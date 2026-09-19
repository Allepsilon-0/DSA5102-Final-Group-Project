# -*- coding: utf-8 -*-
"""贪吃蛇（Snake）强化学习环境，基于 Gymnasium 接口实现。

本文件定义了一个低算力的贪吃蛇环境 `SnakeEnv`，是整个项目的底层。
所有算法（Actor-Critic、DQN、PPO）以及基线策略（随机、教师）都在这个环境上运行。

核心设计（可读性优先）：
- 动作空间：3 个「相对」动作 —— 0=左转、1=右转、2=直行（而不是上下左右）。
- 观测空间：7 通道网格 —— 蛇身 / 食物 / 墙体 / 四个方向，每格取值 0 或 1。
- 奖励函数：支持两种模式（reward_mode），详见 `_calculate_reward`。
- 教师策略：支持两种实现（teacher_mode），详见 `teacher_action`。
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class SnakeEnv(gym.Env[np.ndarray, int]):
    """一个紧凑的贪吃蛇环境，动作是「相对方向」的左转/右转/直行。"""

    # 渲染相关元信息：只支持 human 模式（pygame 窗口），默认帧率 30
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human"],
        "render_fps": 30,
    }
    # 方向 → 观测通道的映射。
    # 注意 direction 用 (row, col) 表示：(-1,0) 上、(1,0) 下、(0,-1) 左、(0,1) 右。
    # 对应 7 通道观测里的第 3/4/5/6 通道。
    direction_channels: ClassVar[dict[tuple[int, int], int]] = {
        (-1, 0): 3,  # 朝上
        (1, 0): 4,   # 朝下
        (0, -1): 5,  # 朝左
        (0, 1): 6,   # 朝右
    }

    def __init__(
        self,
        grid_size: int = 8,
        end_score: int = 8,
        max_steps: int = 300,
        render_mode: str | None = None,
        reward_mode: str = "sparse",
        teacher_mode: str = "shortest_path",
    ) -> None:
        """初始化环境。

        参数:
            grid_size:   棋盘边长（grid_size × grid_size 的方格）。
            end_score:   蛇身长度达到该值即通关（terminated 并给大奖励）。
            max_steps:   单局最大步数，超过则截断（truncated）。
            render_mode: None 不渲染；"human" 用 pygame 窗口渲染。
            reward_mode: "sparse" 稀疏事件奖励（默认，本组改动）；"dense" 密集 shaping（原版对照）。
            teacher_mode:"shortest_path" BFS 最短路径教师（默认，本组改动）；"legacy" 曼哈顿贪心教师（原版对照）。
        """
        super().__init__()
        # ---- 参数合法性校验 ----
        if grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        if end_score < 2:
            raise ValueError("end_score must be at least 2")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be None or 'human'")
        if reward_mode not in {"sparse", "dense"}:
            raise ValueError("reward_mode must be 'sparse' or 'dense'")
        if teacher_mode not in {"shortest_path", "legacy"}:
            raise ValueError("teacher_mode must be 'shortest_path' or 'legacy'")

        # ---- 保存配置 ----
        self.grid_size = grid_size
        self.end_score = end_score
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.teacher_mode = teacher_mode

        # ---- 定义动作空间与观测空间 ----
        # 动作：3 个离散动作（0=左转、1=右转、2=直行）
        self.action_space = spaces.Discrete(3)
        # 观测：7 通道的 grid_size×grid_size 二值网格（取值 0~1）
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(7, grid_size, grid_size),
            dtype=np.float32,
        )

        # ---- 内部状态 ----
        # 蛇身：list，snake[0] 是蛇头，后面依次是身体，每个元素是 (row, col) 的 np.ndarray
        self.snake: list[np.ndarray] = []
        # 食物位置 (row, col)
        self.food_pos = np.zeros(2, dtype=np.int64)
        # 当前移动方向 (row_dir, col_dir)
        self.current_direction = np.array([0, 1], dtype=np.int64)
        self.score = 0          # 本局吃到的食物数（= 蛇身长度 - 1）
        self.step_count = 0     # 本局已走步数
        self.visited: set[tuple[int, int]] = set()            # 历史访问过的格子（dense 模式探索奖励用）
        self.visited_positions: list[np.ndarray] = []         # 蛇头走过的位置序列（绕圈检测用）
        self.game_over = False  # 本局是否已结束（terminated 或 truncated）
        self.window: pygame.Surface | None = None             # pygame 窗口
        self.clock: pygame.time.Clock | None = None           # pygame 时钟
        self.closed = False     # 窗口是否被手动关闭

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境到初始状态，开始新的一局。

        返回:
            (observation, info) —— 初始观测和初始信息字典。
        """
        super().reset(seed=seed)
        del options
        # 蛇从棋盘中心开始，长度 1，朝右（col+1）
        self.snake = [self._get_center_position()]
        self.current_direction = np.array([0, 1], dtype=np.int64)
        # 注意：先设好方向再生成食物，因为 _generate_food 会参考蛇头正前方
        self.food_pos = self._generate_food()
        self.score = 0
        self.step_count = 0
        self.visited.clear()
        self.visited_positions.clear()
        self.game_over = False
        self.closed = False
        observation = self._get_observation()
        if self.render_mode == "human":
            self.render()
        return observation, self._info(executed_action=None)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行一步：按动作移动蛇、更新状态、计算奖励。

        参数:
            action: 0=左转、1=右转、2=直行（相对当前方向的转向）。

        返回:
            (observation, reward, terminated, truncated, info)
            - terminated: 撞墙/撞自己，或蛇长达到 end_score 通关
            - truncated:  步数达到 max_steps 仍未结束（时间截断）
        """
        # 游戏已结束不能再 step
        if self.game_over:
            raise RuntimeError("episode is finished; call reset() before step()")
        # 动作必须是 0/1/2
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}; expected 0, 1, or 2")

        executed_action = int(action)
        # 1) 根据动作计算新方向，再算出新的蛇头位置
        self.current_direction = self._get_direction(executed_action)
        new_head = self.snake[0] + self.current_direction
        self.step_count += 1

        # 2) 判断是否碰撞
        if self._is_collision(new_head):
            reward = -5.0          # 撞墙/撞自己：大惩罚
            terminated = True
        else:
            # 3) 先算「势能/形状」奖励（sparse 下这里基本为 0），再算「吃食物/通关」奖励
            reward = self._calculate_reward(new_head)
            reward_delta, terminated = self._update_snake_and_food(new_head)
            reward += reward_delta

        # 4) 时间截断：步数用尽且没终止
        truncated = self.step_count >= self.max_steps and not terminated
        self.game_over = terminated or truncated
        observation = self._get_observation()
        info = self._info(executed_action=executed_action)
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def teacher_action(self) -> int:
        """返回一个确定性的「安全」动作，供行为克隆或基线对比使用。

        教师策略不会在 step 内部自动改变动作；训练代码需要显式调用本方法拿到动作，
        再把同一个动作传给环境。

        两种实现（由 teacher_mode 决定）：
        - "shortest_path"（默认，本组改动）：对每个动作试走一步，用 BFS 求到食物的
          「真实最短步数」（考虑蛇身障碍），选步数最少的动作；若食物不可达，则退化为
          选「可达自由空间最大」的动作以尽量存活。
        - "legacy"（原版对照）：曼哈顿距离贪心 —— 选「安全且离食物曼哈顿距离最近」的动作。
        """
        if self.teacher_mode == "legacy":
            return self._teacher_action_legacy()

        # ---- shortest_path 模式 ----
        # 第一优先：找能真正走到食物的最短路径
        candidates: list[tuple[int, int]] = []  # (到食物的步数, 动作)
        for action in range(self.action_space.n):
            trial_head = self.snake[0] + self._get_direction(action)  # 试走一步后的蛇头
            if not self._is_safe(trial_head):      # 这一步会撞墙/撞身，跳过
                continue
            steps = self._shortest_path_to_food(trial_head)  # BFS 真实步数，不可达为 None
            if steps is not None:
                candidates.append((steps, action))
        if candidates:
            return min(candidates)[1]   # 选步数最少的动作

        # 食物不可达时：退化为选可达自由空间最大的动作，尽量活得久
        fallback: list[tuple[int, int]] = []  # (可达格子数, 动作)
        for action in range(self.action_space.n):
            trial_head = self.snake[0] + self._get_direction(action)
            if self._is_safe(trial_head):
                fallback.append((self._reachable_space(trial_head), action))
        if fallback:
            return max(fallback)[1]     # 选空间最大的动作
        return 2                        # 实在无路可走，直行（等价于等死）

    def _teacher_action_legacy(self) -> int:
        """原版教师：曼哈顿距离贪心（保留用于对照）。"""

        candidates: list[tuple[float, int]] = []  # (曼哈顿距离, 动作)
        for action in range(self.action_space.n):
            trial_head = self.snake[0] + self._get_direction(action)
            # 既要立即安全，又要 BFS 能到达足够多的格子（避免走进死胡同）
            if self._is_safe(trial_head) and self._bfs_safe_path(trial_head):
                distance = float(np.abs(trial_head - self.food_pos).sum())
                candidates.append((distance, action))
        if not candidates:
            return 2
        return min(candidates)[1]

    def _info(self, executed_action: int | None) -> dict[str, Any]:
        """组装每步/每局返回的 info 字典。"""
        return {
            "score": self.score,           # 当前吃到的食物数
            "length": len(self.snake),     # 蛇身长度
            "steps": self.step_count,      # 已走步数
            "executed_action": executed_action,  # 本步实际执行的动作
        }

    def _is_safe(self, position: np.ndarray) -> bool:
        """判断某个位置是否「立即安全」：不出界，且不撞到蛇身（不含蛇头）。"""
        if np.any(position < 0) or np.any(position >= self.grid_size):
            return False
        return tuple(position) not in map(tuple, self.snake[1:])

    def _bfs_safe_path(self, position: np.ndarray, depth: int = 11) -> bool:
        """BFS 判断从 position 出发能否到达至少 depth 个自由格子（legacy 教师用）。

        原理：把蛇身（除头）当作障碍，做广度优先搜索，统计可达格子数。
        如果可达格子 >= depth，说明周围有足够回旋空间，不容易把自己堵死。
        """
        directions = (
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, -1]),
            np.array([0, 1]),
        )
        queue: deque[np.ndarray] = deque([position])
        blocked = set(map(tuple, self.snake[1:]))   # 障碍 = 蛇身（除头）
        visited = {tuple(position)}
        required_cells = min(depth, self.grid_size * self.grid_size - len(blocked))
        while queue:
            current = queue.popleft()
            if len(visited) >= required_cells:
                return True
            for direction in directions:
                candidate = current + direction
                candidate_key = tuple(candidate)
                in_bounds = np.all(candidate >= 0) and np.all(candidate < self.grid_size)
                if in_bounds and candidate_key not in blocked and candidate_key not in visited:
                    queue.append(candidate)
                    visited.add(candidate_key)
        return False

    def _bfs_distances(self, position: np.ndarray) -> dict[tuple[int, int], int]:
        """BFS 求从 position 到所有可达格的最短距离（蛇身除头视为障碍）。

        这是 shortest_path 教师的核心：返回一个 {格子坐标: 最短步数} 的字典，
        供 `_shortest_path_to_food` 和 `_reachable_space` 使用。
        """
        directions = (
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, -1]),
            np.array([0, 1]),
        )
        blocked = set(map(tuple, self.snake[1:]))   # 障碍 = 蛇身（除头）
        start = tuple(position)
        distances: dict[tuple[int, int], int] = {}
        if start in blocked:
            return distances
        queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])  # (格子, 步数)
        distances[start] = 0
        while queue:
            current, distance = queue.popleft()
            for direction in directions:
                candidate = (
                    current[0] + int(direction[0]),
                    current[1] + int(direction[1]),
                )
                # 越界则跳过
                if not (0 <= candidate[0] < self.grid_size and 0 <= candidate[1] < self.grid_size):
                    continue
                # 撞障碍或已访问则跳过
                if candidate in blocked or candidate in distances:
                    continue
                distances[candidate] = distance + 1
                queue.append((candidate, distance + 1))
        return distances

    def _shortest_path_to_food(self, position: np.ndarray) -> int | None:
        """从 position 到食物的真实最短步数；食物不可达则返回 None。"""
        return self._bfs_distances(position).get(tuple(self.food_pos))

    def _reachable_space(self, position: np.ndarray) -> int:
        """从 position 可达的自由格子数，用于食物不可达时的存活兜底。"""
        return len(self._bfs_distances(position))

    def _get_direction(self, action: int) -> np.ndarray:
        """根据相对动作计算新的移动方向。

        - action 0（左转）：把当前方向逆时针转 90 度。
        - action 1（右转）：把当前方向顺时针转 90 度。
        - action 2（直行）：保持当前方向。
        """
        if action == 0:
            # 左转：(dr, dc) -> (-dc, dr)
            return np.array([-self.current_direction[1], self.current_direction[0]])
        if action == 1:
            # 右转：(dr, dc) -> (dc, -dr)
            return np.array([self.current_direction[1], -self.current_direction[0]])
        return self.current_direction.copy()

    def _calculate_reward(self, new_head: np.ndarray) -> float:
        """计算移动后的「势能/形状」奖励（不含吃食物/通关/碰撞的奖励）。

        两种模式：
        - "sparse"（默认）：纯事件奖励，不做任何距离/探索 shaping，
          只惩罚「原地绕圈」（原地打转是贪吃蛇常见的死法）。
        - "dense"（原版对照）：密集 shaping —— 靠近食物加分、远离扣分、
          访问新格子加分、绕圈扣分。
        """
        if self.reward_mode == "sparse":
            # 稀疏事件奖励：不做距离/探索 shaping，只惩罚原地绕圈
            return -0.5 if self._check_repeated_visit() else 0.0

        # ---- dense 版（原密集 shaping，用于对照） ----
        # 曼哈顿距离：移动前 vs 移动后离食物的距离
        distance_before = np.abs(self.snake[0] - self.food_pos).sum()
        distance_after = np.abs(new_head - self.food_pos).sum()
        reward = 0.2 if distance_after < distance_before else -0.1  # 靠近 +0.2，否则 -0.1

        # 首次访问新格子：+0.2（鼓励探索）
        if tuple(new_head) not in self.visited:
            reward += 0.2
            self.visited.add(tuple(new_head))
        # 绕圈惩罚：-0.4
        if self._check_repeated_visit():
            reward -= 0.4
        return reward

    def _update_snake_and_food(self, new_head: np.ndarray) -> tuple[float, bool]:
        """把蛇头前移一格，处理吃食物/通关，返回 (奖励增量, 是否终止)。"""
        self.snake.insert(0, new_head)   # 蛇头插入到最前面
        reward = 0.0
        if np.array_equal(new_head, self.food_pos):
            # 吃到食物：分数 +1，生成新食物，给大奖励（此时不 pop 蛇尾，蛇变长）
            self.score += 1
            self.food_pos = self._generate_food()
            reward += 10.0
        else:
            # 没吃到：pop 蛇尾（蛇整体前移，长度不变），给一点存活成本
            self.snake.pop()
            reward -= 0.01

        # 记录蛇头走过的位置（绕圈检测用）
        self.visited_positions.append(new_head.copy())
        # 蛇长达到 end_score 即通关
        terminated = len(self.snake) >= self.end_score
        if terminated:
            reward += 100.0
        return reward, terminated

    def _check_repeated_visit(self) -> bool:
        """检测最近几步是否在绕圈（最近 5 步内出现重复位置）。"""
        if len(self.visited_positions) < 5:
            return False
        counts = Counter(map(tuple, self.visited_positions[-5:]))
        return any(count >= 2 for count in counts.values())

    def _get_observation(self) -> np.ndarray:
        """生成 7 通道的观测网格。

        通道含义：
        - 通道 0：蛇身（蛇占的格子为 1）
        - 通道 1：食物（食物格子为 1）
        - 通道 2：墙体（四周一圈边界为 1，内部为 0）
        - 通道 3/4/5/6：当前方向（朝上/下/左/右时，对应整层为 1）
        """
        grid = np.zeros((7, self.grid_size, self.grid_size), dtype=np.float32)
        # 通道 0：蛇身（逐格标记；用标量索引，避免 numpy 把 (r,c) 误当成整行/整列）
        for segment in self.snake:
            grid[0, int(segment[0]), int(segment[1])] = 1.0
        # 通道 1：食物（同样用标量索引，只标单个格子）
        grid[1, int(self.food_pos[0]), int(self.food_pos[1])] = 1.0
        # 通道 2：墙体（先全 1，再把内部 1:-1 区域清零，只留边框）
        grid[2, :, :] = 1.0
        grid[2, 1:-1, 1:-1] = 0.0
        # 通道 3~6：当前方向（整层二值）
        direction_channel = self.direction_channels[tuple(self.current_direction)]
        grid[direction_channel, :, :] = 1.0
        return grid

    def _is_collision(self, position: np.ndarray) -> bool:
        """判断新蛇头是否碰撞：出界，或撞到蛇身（不含蛇头）。"""
        if np.any(position < 0) or np.any(position >= self.grid_size):
            return True
        return tuple(position) in map(tuple, self.snake[1:])

    def _generate_food(self) -> np.ndarray:
        """在空地上随机生成一个新食物。

        本组改动：排除「蛇头正前方一格」，避免食物紧贴蛇头诱导蛇直冲撞墙/撞身。
        若排除后无空地（蛇几乎占满棋盘），则退化为只在蛇身外生成。
        """
        occupied = set(map(tuple, self.snake))   # 蛇身占的格子
        # 蛇头正前方一格也排除（诱导性死亡防护）
        head_ahead = tuple(self.snake[0] + self.current_direction)
        blocked = occupied | {head_ahead}
        free_cells = [
            (row, column)
            for row in range(self.grid_size)
            for column in range(self.grid_size)
            if (row, column) not in blocked
        ]
        if not free_cells:
            # 蛇几乎占满棋盘时退化为只在蛇身外生成
            free_cells = [
                (row, column)
                for row in range(self.grid_size)
                for column in range(self.grid_size)
                if (row, column) not in occupied
            ]
        index = int(self.np_random.integers(len(free_cells)))
        return np.array(free_cells[index], dtype=np.int64)

    def _get_center_position(self) -> np.ndarray:
        """返回棋盘中心坐标（蛇的初始位置）。"""
        return np.array([self.grid_size // 2, self.grid_size // 2], dtype=np.int64)

    def render(self, fps: int | None = None) -> None:
        """用 pygame 渲染当前画面（仅 render_mode="human" 时生效）。"""
        if self.render_mode != "human":
            return
        # 第一次渲染时初始化窗口
        if self.window is None:
            pygame.init()
            self.cell_size = 20
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
            pygame.display.set_caption("RL Snake")
            self.clock = pygame.time.Clock()

        # 处理窗口关闭事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                self.close()
                return

        # 画背景 + 网格线
        self.window.fill((18, 18, 18))
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                rect = pygame.Rect(
                    column * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.window, (48, 48, 48), rect, 1)

        # 画食物（绿色方块）
        food_row, food_column = self.food_pos
        pygame.draw.rect(
            self.window,
            (62, 201, 111),
            pygame.Rect(
                food_column * self.cell_size,
                food_row * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )
        # 画蛇（蛇头红色，身体橙色）
        for index, (row, column) in enumerate(self.snake):
            color = (235, 87, 87) if index == 0 else (224, 132, 79)
            pygame.draw.rect(
                self.window,
                color,
                pygame.Rect(
                    column * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )
        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(fps or self.metadata["render_fps"])

    def close(self) -> None:
        """关闭 pygame 窗口，释放资源。"""
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()
