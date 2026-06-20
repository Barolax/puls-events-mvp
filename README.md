# 🎭 Puls-Events MVP

**Chatbot RAG multi-agents pour la découverte d'événements culturels en France**

[![CI/CD](https://github.com/Barolax/puls-events-mvp/actions/workflows/main.yml/badge.svg)](https://github.com/Barolax/puls-events-mvp/actions)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-multi--agents-green)
![Mistral](https://img.shields.io/badge/Mistral-Large-purple)
![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-orange)

---

## 📌 Présentation

Puls-Events est un assistant conversationnel intelligent qui permet aux utilisateurs de découvrir des événements culturels en France — concerts, expositions, festivals, spectacles — via une interface en langage naturel.

> *"Quels concerts y a-t-il à Lille ce weekend ?"* → Puls recherche, filtre et répond en temps réel avec des recommandations géolocalisées.

---

## 🏗️ Architecture

```
Utilisateur
    ↓
Chainlit UI (Frontend)
    ↓
FastAPI (API REST + Streaming)
    ↓
LangGraph (Orchestrateur multi-agents)
    ├── 1. KoalaAI        → Guardrail input (modération)
    ├── 2. Redis          → Mémoire conversationnelle
    ├── 3. Qdrant + Cohere → RAG + Reranking
    ├── 4. Nominatim      → Géolocalisation
    ├── 5. smolagents     → Recherche web (whitelist)
    ├── 6. Mistral Large  → Génération (streaming, T=0.3)
    └── 7. RAGAS          → Guardrail output (anti-hallucination)
```

---

## 🛠️ Stack technique

| Catégorie | Technologie |
|-----------|-------------|
| Orchestration IA | LangGraph |
| LLM | Mistral Large + Mistral Embed |
| Base vectorielle | Qdrant (670+ événements) |
| Reranking | Cohere Rerank v3 multilingual |
| Mémoire | Redis |
| Guardrail input | KoalaAI Text-Moderation (HuggingFace) |
| Guardrail output | RAGAS faithfulness (seuil 0.2) |
| Recherche web | smolagents + DuckDuckGo + whitelist |
| API | FastAPI (REST + StreamingResponse) |
| Interface | Chainlit (auth + historique conversations) |
| Data quality | Great Expectations |
| Observabilité LLM | Langfuse |
| Monitoring infra | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Conteneurisation | Docker Compose |
| Cloud | AWS ECS Fargate, ECR, ElastiCache, EC2, Secrets Manager |
| Source données | OpenAgenda API (7 agendas régionaux) |

---

## 🚀 Lancement en local

### Prérequis

- Docker + Docker Compose
- Clés API : Mistral, OpenAgenda, HuggingFace, Langfuse, Cohere

### Installation

```bash
# Cloner le repo
git clone https://github.com/Barolax/puls-events-mvp.git
cd puls-events-mvp

# Créer le fichier .env
cp .env.example .env
# Remplir les clés API dans .env

# Lancer tous les services
docker compose up --build -d

# Ingérer les événements dans Qdrant
docker compose exec api python ingestion/vectorizer.py
```

### Accès aux services

| Service | URL |
|---------|-----|
| Interface Chainlit | http://localhost:8001 |
| API FastAPI (docs) | http://localhost:8000/docs |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Qdrant | http://localhost:6333 |

### Comptes de démonstration

| Utilisateur | Mot de passe |
|-------------|-------------|
| `admin` | `admin123` |
| `demo` | `demo123` |

---

## 🧪 Tests

```bash
# Lancer les tests pytest dans Docker
docker compose exec api pytest tests/ -v --tb=short
```

**Résultats CI/CD :** les tests passent sur GitHub Actions à chaque push sur `main`.

---

## 📁 Structure du projet

```
puls-events-mvp/
├── api/                    # FastAPI (endpoints /chat, /chat/stream, /auth)
├── agents/                 # Agents LangGraph
│   ├── graph.py            # Orchestrateur + stream_pipeline()
│   ├── agent_rag.py        # Qdrant + Cohere Rerank
│   ├── agent_memory.py     # Redis
│   ├── agent_geo.py        # Nominatim + Haversine
│   ├── agent_web.py        # smolagents + whitelist
│   ├── llama_guard.py      # KoalaAI guardrail
│   └── ragas_check.py      # RAGAS faithfulness
├── ingestion/              # Pipeline de données OpenAgenda → Qdrant
│   ├── open_agenda.py      # Fetch 7 agendas régionaux
│   ├── chunking.py         # Découpage des événements
│   ├── validation.py       # Great Expectations
│   └── vectorizer.py       # Mistral Embed → Qdrant
├── frontend/               # Chainlit UI
│   ├── app.py              # Auth + streaming + historique
│   └── public/             # Favicon + logos
├── tests/                  # tests pytest
├── monitoring/             # Prometheus + Grafana
├── .github/workflows/      # CI/CD GitHub Actions
└── docker-compose.yml      # Orchestration locale
```

---

## ☁️ Déploiement AWS

L'application est déployée en production sur AWS (région eu-west-3, Paris) :

| Service | Usage |
|---------|-------|
| ECS Fargate | API FastAPI + Frontend Chainlit (serverless) |
| ECR | Registry images Docker (linux/amd64) |
| ElastiCache Redis | Mémoire conversationnelle managée |
| EC2 t3.micro | Instance Qdrant dédiée |
| Secrets Manager | Clés API chiffrées |
| CloudWatch | Logs centralisés |

---

## 📊 Données

- **Source** : OpenAgenda API (open data)
- **Agendas** : Hauts-de-France, Hauts-de-France Musées, Métropole de Lille, Bordeaux, Auvergne-Rhône-Alpes, Île-de-France, Grand Est
- **Volume** : ~667 événements vectorisés
- **Modèle d'embedding** : `mistral-embed` (dimension 1024, distance cosine)

---

## 🔒 Sécurité

- Authentification JWT (FastAPI)
- Modération des inputs (KoalaAI Text-Moderation)
- Détection des hallucinations (RAGAS faithfulness)
- Whitelist de domaines pour la recherche web
- Secrets chiffrés (AWS Secrets Manager)

