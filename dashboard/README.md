# Dashboard TP Intelligence Artificielle 2
## DIPES 2 — 4eme annee (S2) — 2025-2026

Dashboard Flask de visualisation des resultats du TP Deep Learning.

---

## Installation et lancement

### 1. Installer les dependances

```bash
pip install flask
```

### 2. Organiser vos fichiers

Placez le dossier `dashboard/` au meme niveau que vos dossiers de sorties :

```
votre_projet/
    dashboard/          <- ce dossier
        app.py
        templates/
        requirements.txt
    outputs_tp1/        <- genere par dashboard_census.py
        01_analyse_exploratoire.png
        02_correlation_matrix.png
        ...
    outputs_tp2/        <- genere par le notebook part2__2_.ipynb
        ...
    dashboard_census.py
    part2__2_.ipynb
```

### 3. Lancer le serveur

```bash
cd dashboard
python app.py
```

Puis ouvrir dans votre navigateur : **http://localhost:5050**

---

## Structure du Dashboard

| Page              | URL          | Contenu                                              |
|-------------------|--------------|------------------------------------------------------|
| Accueil           | /            | Vue d'ensemble, navigation, contexte des datasets    |
| Partie 1          | /partie1     | Toutes les figures Census Income par section         |
| Partie 2          | /partie2     | Figures Bank Telemarketing + Fashion MNIST           |
| Methodologie      | /methodes    | Explication des algorithmes et choix metodologiques  |
| Guide Presentation| /guide       | Checklist, plan de presentation, conclusions         |

---

## Fonctionnalites

- **Navigation multi-pages** avec menu persistent
- **Lightbox** : cliquez sur une figure pour l'agrandir en plein ecran
- **Onglets interactifs** dans la Partie 2 (Bank / MNIST / Toutes)
- **Tableaux comparatifs** des performances des modeles
- **Checklist** des elements a rendre pour le TP
- **Descriptions contextuelles** de chaque figure

---

## Notes

- Si les dossiers `outputs_tp1/` ou `outputs_tp2/` sont absents, une alerte s'affiche.
- Les chemins sont detectes automatiquement (relatifs et absolus).
- Le port par defaut est 5050 (modifiable dans app.py ligne finale).
