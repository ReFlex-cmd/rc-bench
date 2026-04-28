import io
from pathlib import Path
from typing import List

from datetime import timedelta
from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.responses import Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from jose import JWTError, jwt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Импорты наших модулей
from rc_bench.database import get_db
from rc_bench.config import settings
from rc_bench.core.security import get_password_hash, verify_password, create_access_token
from rc_bench.models import User, Experiment, ExperimentStatus, Result
from rc_bench.schemas import ExperimentCreate, ExperimentRead, UserCreate, UserRead, Token, TokenData
from rc_bench.tasks import run_experiment_task

app = FastAPI(title="RC-Bench API", version="0.1.0")

# URL для получения токена (Swagger использует его для кнопки Authorize)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --------------------------------------------------------------------------
# DEPENDENCY: Получение текущего пользователя
# --------------------------------------------------------------------------
async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
        
    query = select(User).where(User.email == token_data.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
        
    return user

@app.get("/")
async def root():
    return {"message": "Welcome to RC-Bench Platform"}

# --------------------------------------------------------------------------
# AUTH: Регистрация
# --------------------------------------------------------------------------
@app.post("/register", response_model=UserRead)
async def register_user(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    query = select(User).where(User.email == user_data.email)
    result = await db.execute(query)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = get_password_hash(user_data.password)
    new_user = User(email=user_data.email, hashed_password=hashed_pw)
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

# --------------------------------------------------------------------------
# AUTH: Логин (Выдача токена)
# --------------------------------------------------------------------------
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: AsyncSession = Depends(get_db)
):
    query = select(User).where(User.email == form_data.username)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --------------------------------------------------------------------------
# EXPERIMENTS: Создание (ЗАЩИЩЕНО)
# --------------------------------------------------------------------------
@app.post("/experiments/", response_model=ExperimentRead)
async def create_experiment(
    experiment_data: ExperimentCreate, 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    new_experiment = Experiment(
        reservoir_type=experiment_data.reservoir.type,
        dataset_name=experiment_data.dataset.name,
        config=experiment_data.model_dump(),
        status=ExperimentStatus.QUEUED,
        owner_id=current_user.id
    )
    
    db.add(new_experiment)
    await db.commit()
    
    # --- ИСПРАВЛЕНИЕ ---
    # Вместо db.refresh, который не умеет грузить связи (results),
    # мы делаем явную выборку с selectinload.
    # Это корректно загрузит пустой список results без ошибок MissingGreenlet.
    query = (
        select(Experiment)
        .options(selectinload(Experiment.results))  # <-- Грузим связи сразу
        .where(Experiment.id == new_experiment.id)
    )
    result = await db.execute(query)
    loaded_experiment = result.scalar_one()
    
    # Отправляем задачу в Celery
    run_experiment_task.delay(loaded_experiment.id)
    
    return loaded_experiment
    
# --------------------------------------------------------------------------
# EXPERIMENTS: Список (ЗАЩИЩЕНО)
# --------------------------------------------------------------------------
@app.get("/experiments/", response_model=List[ExperimentRead])
async def list_experiments(
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = (
        select(Experiment)
        .options(selectinload(Experiment.results))
        .offset(skip)
        .limit(limit)
        .order_by(Experiment.id.desc())
    )
    result = await db.execute(query)
    return result.scalars().all()

# --------------------------------------------------------------------------
# EXPERIMENTS: Сравнение (до /{experiment_id}, иначе "compare" -> id)
# --------------------------------------------------------------------------
@app.get("/experiments/compare")
async def compare_experiments(
    ids: str = Query(..., description="ID экспериментов через запятую"),
    format: str = Query("json", description="Формат ответа: json или csv"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        id_list = [int(x.strip()) for x in ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="ids must be comma-separated integers")

    query = (
        select(Experiment)
        .options(selectinload(Experiment.results))
        .where(Experiment.id.in_(id_list))
    )
    result = await db.execute(query)
    experiments = result.scalars().all()

    if not experiments:
        raise HTTPException(status_code=404, detail="No experiments found")

    rows = []
    for exp in experiments:
        row = {
            "id": exp.id,
            "reservoir_type": exp.reservoir_type,
            "dataset_name": exp.dataset_name,
            "status": exp.status,
        }
        if exp.results:
            metrics = (exp.results[-1].result_data.get("metrics") or {})
            row.update({
                "nrmse_range": metrics.get("nrmse_range"),
                "nrmse_std": metrics.get("nrmse_std"),
                "nrmse_var": metrics.get("nrmse_var"),
                "rmse": metrics.get("rmse"),
                "mse": metrics.get("mse"),
                "mae": metrics.get("mae"),
                "prediction_horizon": metrics.get("prediction_horizon"),
                "val_nrmse_range": metrics.get("val_nrmse_range"),
                "train_time": metrics.get("train_time"),
                "inference_latency": metrics.get("inference_latency"),
                "peak_memory": metrics.get("peak_memory"),
            })
        rows.append(row)

    if format == "csv":
        if not rows:
            return Response(content="", media_type="text/csv")
        header = ",".join(rows[0].keys())
        lines = [header] + [",".join(str(v) for v in r.values()) for r in rows]
        return Response(content="\n".join(lines), media_type="text/csv")

    return rows

# --------------------------------------------------------------------------
# EXPERIMENTS: Детали
# --------------------------------------------------------------------------
@app.get("/experiments/{experiment_id}", response_model=ExperimentRead)
async def get_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(Experiment)
        .options(selectinload(Experiment.results))
        .where(Experiment.id == experiment_id)
    )
    result = await db.execute(query)
    experiment = result.scalar_one_or_none()

    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return experiment

# --------------------------------------------------------------------------
# EXPERIMENTS: График предсказаний
# --------------------------------------------------------------------------
@app.get("/experiments/{experiment_id}/plot")
async def plot_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = (
        select(Result)
        .where(Result.experiment_id == experiment_id)
        .order_by(Result.id.desc())
    )
    result = await db.execute(query)
    res = result.scalar_one_or_none()

    if res is None:
        raise HTTPException(status_code=404, detail="Result not found")

    artifact_paths = (res.result_data or {}).get("artifact_paths", {})
    preds_path = artifact_paths.get("preds")
    y_test_path = artifact_paths.get("y_test")

    if not preds_path or not y_test_path:
        raise HTTPException(status_code=404, detail="Prediction files not found in result_data")

    preds_file = Path(preds_path)
    y_test_file = Path(y_test_path)

    if not preds_file.exists() or not y_test_file.exists():
        raise HTTPException(status_code=404, detail="Prediction files missing on disk")

    preds = np.load(preds_file)
    y_test = np.load(y_test_file)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_test, label="y_test", linewidth=0.8)
    ax.plot(preds, label="prediction", linewidth=0.8, alpha=0.8)
    ax.set_title(f"Experiment {experiment_id}")
    ax.set_xlabel("t")
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")
