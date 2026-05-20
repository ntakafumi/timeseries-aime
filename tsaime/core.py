# ts_aime/core.py
# Time-Series AIME (rolling AIME for time series) on top of aime-xai and pyEDM.
# Author: (your names)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

try:
    # public package you already released
    from aime_xai import AIME  # pip install aime-xai
except Exception as e:
    raise ImportError("aime-xai が必要です。`pip install aime-xai` を実行してください。") from e


def _zscore(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mu = df.mean()
    sd = df.std(ddof=0).replace(0, 1.0)
    return (df - mu) / sd, mu, sd


@dataclass
class RollingAIMEConfig:
    # 予測・窓
    window: int = 90
    tp: int = 1

    # 置換(帰無)の設定
    R: int = 0               # 0 なら帰無を計算しない（エンベロープ/ p値を返さない）
    seed: int = 0
    preserve_weekly: bool = False  # 週保存円周シフト（block_null が None のときのみ有効）

    # ブロック帰無： 'month' | 'quarter' | 'season' | None
    # 月/四半期/季節ごとにブロック（連続部分列）をランダム順に並べ替える
    block_null: Optional[str] = None
    date_column: Optional[str] = None  # 窓 DataFrame 内の datetime64 列名（例: "Date"）

    # 数値安定化
    ridge: float = 1e-12
    pre_normalize_y: bool = False  # 通常 False（aime-xai の normalize=True に委ねる）


class RollingAIME:
    """
    Rolling AIME on top of pyEDM (S-Map) + aime-xai.

    - 窓内 S-Map: Time(t) -> yhat(t+Tp)
    - 整列: 特徴 X(t) と 予測 yhat(t+Tp) を Time で厳密に inner join
    - AIME: aime-xai の A_dagger を窓ごとに作成（normalize=True）
    - 置換帰無:
        * None          : 円周シフト（標準）
        * preserve_weekly=True : 7日の倍数だけ円周シフト（曜日保存）
        * block_null in {month, quarter, season}: ブロック入れ替え（date_column 必須）

    returns:
        global_ts:  Time, features（グローバルAIME）
        local_ts:   Time, features（窓末尾ローカルAIME）
        env_lo/hi:  95%帯（無指定なら None）
        pvals:      二側 p 値（add-one 補正）
    """

    def __init__(self, cfg: RollingAIMEConfig):
        self.cfg = cfg
        self.last_A_dagger_: Optional[np.ndarray] = None  # (p,1) of last window

    # ---------- pyEDM で学習・予測（窓内） ----------
    @staticmethod
    def _fit_smap_and_predict(sub: pd.DataFrame,
                              feats: List[str],
                              target: str,
                              E: int,
                              theta: float,
                              embedded: bool,
                              tp: int) -> pd.DataFrame:
        try:
            import pyEDM as ed
        except Exception as e:
            raise ImportError("pyEDM が必要です。`pip install pyEDM==1.14.0.2` 等を実行してください。") from e

        cols_str = " ".join(feats)
        # target が feats に含まれていなければ、DataFrame には同梱する
        df_for_edm = sub[["Time"] + feats + ([target] if target not in feats else [])]

        sm = ed.SMap(
            dataFrame=df_for_edm,
            columns=cols_str, target=target,
            lib=f"1 {len(sub)}", pred=f"1 {len(sub)}",
            E=(len(feats) if embedded else E),
            embedded=embedded, theta=theta, Tp=tp, showPlot=False
        )
        pred = sm["predictions"][["Time", "Observations", "Predictions"]].copy()
        return pred  # Time is t+tp

    # ---------- 整列: X(t) と yhat(t+tp) を一致 ----------
    @staticmethod
    def _align_xy_for_aime(sub: pd.DataFrame, pred: pd.DataFrame,
                           feats: List[str], tp: int) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series]:
        Xw = sub[["Time"] + feats].copy()
        Xw["Time_pred"] = Xw["Time"] + tp  # t -> t+tp
        P = pred[["Time", "Predictions"]].rename(columns={"Time": "Time_pred"})
        merged = Xw.merge(P, on="Time_pred", how="inner", validate="one_to_one")

        X_al = merged[feats].copy()
        yhat = merged["Predictions"].to_numpy().astype(float)
        Xz, _, _ = _zscore(X_al)

        t_pred = merged["Time_pred"]  # t+tp
        t_src  = merged["Time"]       # t
        return Xz, yhat, t_pred, t_src

    # ---------- AIME（aime-xai） ----------
    def _aime_from_window(self, Xz: pd.DataFrame, yhat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        returns:
          A_dagger: (p,1)
          gi:       (p,)  global importance
          l_last:   (p,)  local importance at the last row (Hadamard)
        """
        Xn = np.asarray(Xz, dtype=float)
        y  = np.asarray(yhat, dtype=float).reshape(-1, 1)

        if self.cfg.pre_normalize_y:
            mu = y.mean()
            sd = y.std(ddof=0) if y.std(ddof=0) > 0 else 1.0
            y = (y - mu) / sd

        expl = AIME()
        # AIME 側で X,y を正規化して A† を作成（AIME の前提）
        expl.create_explainer(Xn, y, normalize=False)
        A_dagger = np.asarray(expl.A_dagger, dtype=float)  # (p,1)
        self.last_A_dagger_ = A_dagger

        gi = A_dagger.flatten()
        l_series = (y * A_dagger.T) * Xn  # (n,p)
        l_last   = l_series[-1, :]
        return A_dagger, gi, l_last

    # ---------- ブロックラベルの作成 ----------
    @staticmethod
    def _season_from_month(m: int) -> str:
        # DJF/ MAM/ JJA/ SON
        if m in (12, 1, 2):
            return "DJF"
        if m in (3, 4, 5):
            return "MAM"
        if m in (6, 7, 8):
            return "JJA"
        return "SON"

    def _make_block_labels(self, sub: pd.DataFrame) -> Optional[np.ndarray]:
        """
        block_null が None でなければ、サンプルごとのブロックラベル（同長の配列）を返す。
        date_column が無ければエラー。
        """
        mode = (self.cfg.block_null or "").lower()
        if mode not in ("month", "quarter", "season"):
            return None

        if self.cfg.date_column is None or self.cfg.date_column not in sub.columns:
            raise ValueError(
                "季節ブロック帰無を使うには、窓 DataFrame に datetime64 の `date_column` を含めてください。"
            )
        dt = pd.to_datetime(sub[self.cfg.date_column])
        if mode == "month":
            labels = dt.dt.to_period("M").astype(str).to_numpy()
        elif mode == "quarter":
            labels = (dt.dt.year.astype(str) + "Q" + dt.dt.quarter.astype(str)).to_numpy()
        else:  # season
            labels = np.array([f"{y}-{self._season_from_month(m)}" for y, m in zip(dt.dt.year, dt.dt.month)], dtype=object)
        return labels

    # ---------- 置換（円周/週保存/ブロック）で帰無帯と p 値 ----------
    def _perm_envelope_and_p(
        self,
        sub: pd.DataFrame,
        Xz: pd.DataFrame,
        yhat: np.ndarray,
        gi_obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        1窓分の 95%帯と p 値（二側）。X は正規化済み（DataFrame）、yhat は 1D。
        観測グローバル寄与 gi_obs と比較して p 値を計算（add-one 補正）。
        """
        R = int(self.cfg.R)
        if R <= 0:
            return np.array([]), np.array([]), np.array([])

        rng = np.random.default_rng(self.cfg.seed)
        Xn  = np.asarray(Xz, dtype=float)
        y   = np.asarray(yhat, dtype=float).ravel()

        # z-score 固定
        mu = y.mean(); sd = y.std(ddof=0); y = (y - mu) / (sd if sd > 0 else 1.0)
        n, p = Xn.shape
        gi_perm = np.zeros((R, p))

        # ブロックラベル（あればブロック帰無、無ければ円周シフト）
        labels = self._make_block_labels(sub)

        if labels is None:
            # 円周シフト（ preserve_weekly=True なら 7 の倍数シフト ）
            for r in range(R):
                if self.cfg.preserve_weekly:
                    kmax = max(1, n // 7)
                    shift = 7 * rng.integers(1, kmax + 1)
                else:
                    shift = rng.integers(1, n)
                y_perm = np.roll(y, shift)
                denom  = float(np.dot(y_perm, y_perm)) + self.cfg.ridge
                A_dag_p = (Xn.T @ y_perm.reshape(-1, 1)) / denom  # (p,1)
                gi_perm[r, :] = A_dag_p.flatten()
        else:
            # 季節ブロック帰無：同一ラベルの連続ブロックを抽出してシャッフル
            # 手順：
            # 1) ラベルごとにインデックス連続区間（runs）を検出
            # 2) run のリストをランダム順に並べ替え
            # 3) 連結して新しい y_perm を作成（ブロック内部の順序は保持）
            idx = np.arange(n)
            runs: List[np.ndarray] = []
            start = 0
            for i in range(1, n + 1):
                if i == n or labels[i] != labels[i - 1]:
                    runs.append(idx[start:i])
                    start = i
            for r in range(R):
                order = rng.permutation(len(runs))
                new_idx = np.concatenate([runs[k] for k in order])
                y_perm = y[new_idx]
                denom  = float(np.dot(y_perm, y_perm)) + self.cfg.ridge
                A_dag_p = (Xn.T @ y_perm.reshape(-1, 1)) / denom
                gi_perm[r, :] = A_dag_p.flatten()

        lo = np.quantile(gi_perm, 0.025, axis=0)
        hi = np.quantile(gi_perm, 0.975, axis=0)
        # add-one 補正付き二側 p 値
        count = np.sum(np.abs(gi_perm) >= np.abs(gi_obs.reshape(1, -1)), axis=0)
        pvals = (count + 1.0) / (R + 1.0)
        return lo, hi, pvals

    # ---------- 外部API：pyEDMで rolling ----------
    def fit_with_pyEDM(self,
                       df: pd.DataFrame,
                       columns: List[str],
                       target: str,
                       E: int,
                       theta: float,
                       embedded: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        df: 必須列 = ['Time'] + features（Time は 1..N の整数）
            ※ 季節ブロック帰無を使う場合は、さらに datetime64 の `date_column` を含めてください。
        returns:
          global_ts, local_ts, env_lo, env_hi, pvals
        """
        feats = list(columns)
        win   = self.cfg.window
        tp    = self.cfg.tp

        G_rows: List[Dict] = []
        L_rows: List[Dict] = []
        ENV_LO: List[Dict] = []
        ENV_HI: List[Dict] = []
        PV:     List[Dict] = []

        for end in range(win, len(df) + 1):
            sub = df.iloc[end - win:end].copy()

            # 1) 窓内 S-Map → 予測（Time は t+tp）
            pred = self._fit_smap_and_predict(sub, feats, target, E=E, theta=theta, embedded=embedded, tp=tp)
            # 2) 整列（X(t) と yhat(t+tp)）
            Xz, yhat, t_pred, _ = self._align_xy_for_aime(sub, pred, feats, tp=tp)
            if len(yhat) == 0:
                continue

            # 3) AIME（aime-xai で A†）
            A_dag, gi, l_last = self._aime_from_window(Xz, yhat)
            t_end_pred = int(t_pred.iloc[-1])

            G_rows.append(dict(Time=t_end_pred, **{f: v for f, v in zip(feats, gi)}))
            L_rows.append(dict(Time=t_end_pred, **{f: l_last[j] for j, f in enumerate(feats)}))

            # 4)（任意）置換帯と p 値
            if self.cfg.R > 0:
                lo, hi, p = self._perm_envelope_and_p(sub, Xz, yhat, gi_obs=gi)
                ENV_LO.append(dict(Time=t_end_pred, **{f: lo[j] for j, f in enumerate(feats)}))
                ENV_HI.append(dict(Time=t_end_pred, **{f: hi[j] for j, f in enumerate(feats)}))
                PV.append(    dict(Time=t_end_pred, **{f: p[j]  for j, f in enumerate(feats)}))

        global_ts = pd.DataFrame(G_rows)
        local_ts  = pd.DataFrame(L_rows)
        env_lo = pd.DataFrame(ENV_LO) if ENV_LO else None
        env_hi = pd.DataFrame(ENV_HI) if ENV_HI else None
        pvals  = pd.DataFrame(PV)     if PV     else None
        return global_ts, local_ts, env_lo, env_hi, pvals
