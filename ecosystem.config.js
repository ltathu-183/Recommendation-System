module.exports = {
  apps: [
    {
      name: "rec-sys-train",
      cwd: "/home/ltathu183/rec_sys",
      script: "uv",
      args: "run python scripts/train.py --config configs/default.yaml --output artifacts/two_stage_lgbm.pkl",
      autorestart: false,
      watch: false,
      max_restarts: 0,
      time: true,
    },
    {
      name: "rec-sys-ensemble",
      cwd: "/home/ltathu183/rec_sys",
      script: "uv",
      args: "run python -m rec_sys.ensemble",
      autorestart: false,
      watch: false,
      max_restarts: 0,
      time: true,
    },
    {
      name: "rec-sys-evaluate",
      cwd: "/home/ltathu183/rec_sys",
      script: "uv",
      args: "run python scripts/evaluate.py --config configs/default.yaml",
      autorestart: false,
      watch: false,
      max_restarts: 0,
      time: true,
    },
  ],
};
