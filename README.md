# BLIPollama – Image Captioning (BLIP & Ollama)

BLIPollama est un outil Python permettant de générer automatiquement des **descriptions d’images (captions)** à partir d’un répertoire local, en utilisant soit :

- **BLIP** (via Hugging Face Transformers, exécution locale CPU/GPU)
- **Ollama** (via un modèle multimodal local comme `llava`)

L’architecture est volontairement **propre et modulaire** :
- le service est agnostique des modèles
- les backends (BLIP / Ollama) sont injectés
- une CLI permet de choisir facilement le moteur

---

## ✨ Fonctionnalités

- 📁 Analyse d’un répertoire d’images
- 🖼️ Filtres par extensions (`jpg`, `png`, …)
- 🔁 Mode récursif optionnel
- 🧠 Choix du backend : BLIP ou Ollama
- 📊 Barre de progression (`tqdm`)
- 🧪 Tests unitaires (`pytest`)
- 📦 Structure prête pour un vrai package Python

---

## 🧱 Prérequis

- Python **3.10+**
- (Optionnel) GPU CUDA si utilisation de BLIP avec accélération
- (Optionnel) Ollama installé et lancé localement

---

## 📦 Installation

### 1️⃣ Cloner le projet
```bash
git clone <ton-repo-git>
cd IBM
2️⃣ Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate
3️⃣ Installer les dépendances
pip install -r requirements.txt
⚠️ Remarque :
Le package torch peut nécessiter une installation spécifique selon ton GPU / CUDA.
Réf. : https://pytorch.org/get-started/locally/

🧪 Vérifier l’installation
Lancer les tests unitaires
pytest
Résultat attendu :

1 passed in X.XXs
🚀 Utilisation (CLI)
Exemple avec BLIP
python main.py \
  --backend blip \
  --dir /mnt/d/Photos/100MEDIA \
  --ext jpg jpeg png \
  --out captions_blip.txt
Exemple avec Ollama
python main.py \
  --backend ollama \
  --dir /mnt/d/Photos/100MEDIA \
  --ext jpg png \
  --ollama-model llava \
  --out captions_ollama.txt
Options principales
Option	Description
--backend	blip ou ollama
--dir	Répertoire d’images à analyser
--ext	Extensions à inclure
--recursive	Analyse récursive
--out	Fichier de sortie
🧠 Architecture du projet
IBM/
├─ blipollama/
│  ├─ __init__.py
│  ├─ models.py        # Protocol + dataclasses
│  └─ service.py       # Orchestrateur (agnostique)
├─ tests/
│  └─ test_service_minimal.py
├─ main.py             # CLI & configuration
├─ requirements.txt
├─ pytest.ini
└─ README.md
Principe clé
Le service ne connaît pas les modèles
Il reçoit un objet capable de produire une caption (caption(Path) -> str)

🧩 Ajouter un nouveau backend
Pour ajouter un moteur (ex: Florence, API externe, etc.) :

class MyBackend:
    name = "mybackend"

    def caption(self, image_path: Path) -> str:
        return "my caption"
Puis l’injecter dans VisionCaptionService.

Aucune modification du service n’est nécessaire.

🧹 Git & bonnes pratiques
Certains fichiers sont volontairement ignorés :

environnements virtuels

caches Python / pytest

fichiers générés (captions_*.txt)

images locales

Voir .gitignore.

📌 Notes
BLIP offre des captions plus structurées et précises

Ollama est plus simple à déployer si tu as déjà un stack local LLM

Le projet est prêt pour :

un fallback automatique BLIP → Ollama

une intégration RAG

une base de données (SQL / vectorielle)

📄 Licence
Projet pédagogique / expérimental.
À adapter selon ton usage (personnel / pro).

🙌 Auteur
Patrick Vandervoort
(Projet Coursera / IBM – Vision & LLM)
