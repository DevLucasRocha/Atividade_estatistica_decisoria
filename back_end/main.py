from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from typing import List
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Estatística Descritiva API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Em produção, coloque a URL do seu front
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para entrada de dados por classe
class ClasseFrequencia(BaseModel):
    limite_inferior: float
    limite_superior: float
    frequencia: int

    @model_validator(mode='after')
    def validar_limites(self):
        if self.limite_superior <= self.limite_inferior:
            raise ValueError("O limite superior deve ser maior que o limite inferior")
        if self.frequencia < 0:
            raise ValueError("A frequência não pode ser negativa")
        return self

class DadosEntrada(BaseModel):
    classes: List[ClasseFrequencia]

@app.post("/calcular-estatisticas")
async def calcular_estatisticas(entrada: DadosEntrada):
    classes = entrada.classes
    if not classes:
        raise HTTPException(status_code=400, detail="Adicione pelo menos uma classe")

    n = sum(c.frequencia for c in classes)
    
    # 1. Cálculos de Ponto Médio (xi) e Frequência Acumulada (fac)
    fac_acumulada = 0
    lista_xi = []
    lista_fi = []
    lista_fac = []
    
    for c in classes:
        xi = (c.limite_inferior + c.limite_superior) / 2
        fac_acumulada += c.frequencia
        lista_xi.append(xi)
        lista_fi.append(c.frequencia)
        lista_fac.append(fac_acumulada)

    # --- MÉDIA ---
    media = sum(f * x for f, x in zip(lista_fi, lista_xi)) / n

    # --- MEDIANA (Lógica da imagem da aula) ---
    posicao_mediana = n / 2
    # Encontrar a classe da mediana
    idx_med = next(i for i, f in enumerate(lista_fac) if f >= posicao_mediana)
    
    l_inf_med = classes[idx_med].limite_inferior
    fac_ant = lista_fac[idx_med - 1] if idx_med > 0 else 0
    fi_med = lista_fi[idx_med]
    h = classes[idx_med].limite_superior - classes[idx_med].limite_inferior
    
    mediana = l_inf_med + ((posicao_mediana - fac_ant) / fi_med) * h

    # --- MODA (Classe Modal) ---
    idx_moda = lista_fi.index(max(lista_fi))
    moda = lista_xi[idx_moda]

    # --- VARIÂNCIA E DESVIO PADRÃO ---
    soma_quadrados = sum(f * ((x - media) ** 2) for f, x in zip(lista_fi, lista_xi))
    variancia = soma_quadrados / (n - 1) if n > 1 else 0
    desvio_padrao = np.sqrt(variancia)

    return {
        "n_total": n,
        "media": round(media, 2),
        "mediana": round(mediana, 2),
        "moda_bruta": round(moda, 2),
        "variancia": round(variancia, 2),
        "desvio_padrao": round(desvio_padrao, 2)
    }