FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV SDL_VIDEODRIVER=dummy
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN python -m pip install --upgrade pip
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install numpy pandas matplotlib seaborn gymnasium pygame tensorboard jupyterlab nbconvert ipykernel

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=snake"]
