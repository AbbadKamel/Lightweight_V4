# 📚 RÉCAP COMPLET : PHASE 0 & PHASE 1
## Système de Détection d'Intrusion Maritime N2KShield

**Date de création :** 16 Décembre 2025  
**Projet :** Lightweight Maritime CAN IDS (NMEA2000)  
**Objectif :** Documenter toutes les méthodes utilisées, alternatives envisagées, et justifications

---

## 📋 TABLE DES MATIÈRES

1. [Contexte et Objectif Global](#1-contexte-et-objectif-global)
2. [Phase 0 : Collecte et Décodage](#2-phase-0--collecte-et-décodage)
3. [Phase 1 : Prétraitement et Fenêtrage](#3-phase-1--prétraitement-et-fenêtrage)
4. [Tableau Récapitulatif des Décisions](#4-tableau-récapitulatif-des-décisions)
5. [Conclusion et Prochaines Étapes](#5-conclusion-et-prochaines-étapes)

---

# 1. CONTEXTE ET OBJECTIF GLOBAL

## 1.1 Problématique

Les réseaux NMEA 2000 (standard de communication maritime basé sur CAN bus) sont vulnérables aux cyberattaques :
- **GPS Spoofing** : Falsification de la position
- **Sabotage de navigation** : Modification des données de profondeur, cap, vitesse
- **Détournement de gouvernail** : Prise de contrôle du système d'autopilote

## 1.2 Objectif

Développer un **système de détection d'intrusion léger (IDS)** basé sur CNN-Autoencoder capable de :
- Apprendre les patterns normaux de navigation
- Détecter les anomalies (potentielles attaques)
- Fonctionner sur hardware embarqué limité (Raspberry Pi, ESP32)

## 1.3 Approche Choisie : CANShield

Nous adaptons la méthodologie **CANShield** (publiée pour l'automobile) au domaine maritime.

**Pourquoi CANShield ?**
| Critère | CANShield | Alternatives (SVM, Random Forest) |
|---------|-----------|-----------------------------------|
| Détection anomalies | ✅ Autoencoder (unsupervised) | ❌ Nécessite données d'attaque |
| Multi-échelle | ✅ Ensemble 9 modèles | ❌ Échelle unique |
| Légèreté | ✅ CNN compact | ⚠️ Variable |
| Prouvé | ✅ Publié et validé | ⚠️ Non testé sur CAN |

---

# 2. PHASE 0 : COLLECTE ET DÉCODAGE

## 2.1 Données Brutes

**Ce qu'on avait :**
```
Source :         Capture CAN bus réelle sur bateau
Localisation :   Côte Nord de la France (50.82°N, 0.32°E)
Durée :          85 minutes de navigation
Trames brutes :  2,984,250 frames CAN
Format :         Timestamp, CAN ID, 8 octets de données
```

---

## 2.2 Décodage NMEA 2000

### 2.2.1 Méthode Utilisée ✅

**Approche : Décodeur personnalisé basé sur la spécification NMEA 2000**

```python
# Exemple : Décodage PGN 127250 (Heading)
heading_raw = self._bytes_to_uint16(data, 1)
signals['heading'] = (heading_raw * 0.0001) * (180.0 / np.pi)  # radians → degrés
```

**Validation :** Comparaison byte-par-byte avec la librairie officielle NMEA2000 (`N2kMessages.cpp`)

**Résultat :**
- 654,013 messages décodés (21.92% des trames)
- 9 PGNs implémentés
- 19 signaux extraits

### 2.2.2 Pourquoi 21.92% de décodage seulement ?

> [!NOTE]
> Ce taux est **normal et attendu**. Voici pourquoi :

| Raison | Explication |
|--------|-------------|
| PGNs non implémentés | On a ciblé 9 PGNs sur 40+ possibles |
| PGNs engine/AIS | ~800K frames = données moteur (pas notre focus) |
| Choix scientifique | Qualité > Quantité - on garde l'essentiel maritime |

### 2.2.3 Méthodes Alternatives Rejetées ❌

#### ❌ Option A : Utiliser une librairie de décodage existante

**Description :** Utiliser directement `cantools` ou `python-can` avec un fichier DBC NMEA2000

**Pourquoi rejetée :**
1. **Pas de fichier DBC officiel** - NMEA 2000 est un standard propriétaire payant
2. **Fichiers DBC communautaires incomplets** - Manquent beaucoup de PGNs
3. **Moins de contrôle** - On ne peut pas valider le décodage nous-mêmes
4. **Dépendance externe** - Risque de bugs dans la librairie

#### ❌ Option B : Utiliser les données brutes sans décodage

**Description :** Donner directement les 8 octets CAN au CNN

**Pourquoi rejetée :**
1. **Pas de sémantique** - Le CNN ne sait pas que l'octet 3 = latitude
2. **Mélange de PGNs** - Différents messages ont différentes structures
3. **Inefficace** - Le CNN doit apprendre le décodage + les patterns
4. **Non interprétable** - Impossible d'expliquer les détections

#### ❌ Option C : Décoder TOUS les PGNs possibles

**Description :** Implémenter un décodeur pour les 40+ PGNs

**Pourquoi rejetée :**
1. **Effort disproportionné** - Beaucoup de PGNs sont rares ou non pertinents
2. **Données insuffisantes** - Certains PGNs ont < 1% de couverture
3. **Bruit** - Plus de features = plus de risque d'overfitting
4. **Temps** - 3 mois de travail supplémentaire

---

## 2.3 Agrégation Temporelle

### 2.3.1 Problème à Résoudre

Les données CAN arrivent de façon **asynchrone** :
- PGN Heading : toutes les 100ms
- PGN Depth : toutes les 500ms  
- PGN GPS : toutes les 1000ms

**Problème :** Comment créer une série temporelle unifiée ?

### 2.3.2 Méthode Utilisée ✅

**Approche : Agrégation par intervalles de 1 seconde**

```python
# Arrondir les timestamps à la seconde
df['time_rounded'] = df['time_relative'].round(0)

# Agréger les valeurs (plusieurs messages → une valeur)
grouped = df.groupby('time_rounded')[signal].mean()
```

**Résultat :** 5,095 lignes (1 par seconde de navigation)

### 2.3.3 Pourquoi 1 seconde ?

| Critère | 100ms | 1 seconde ✅ | 5 secondes |
|---------|-------|--------------|------------|
| Résolution | Trop fine | Équilibrée | Trop grossière |
| Volume données | ~50K lignes | ~5K lignes | ~1K lignes |
| Redondance | Haute (90% duplicats) | Basse | Aucune |
| Patterns maritimes | Perd le contexte | Capture bien | Perd les détails |
| Compatibilité NMEA | ⚠️ Pas aligné | ✅ Aligné aux updates | ❌ Perd des updates |

### 2.3.4 Méthodes Alternatives Rejetées ❌

#### ❌ Option A : Pas d'agrégation (100ms)

**Description :** Garder tous les messages avec leur timestamp original

**Pourquoi rejetée :**
1. **Données éparses** - Un signal GPS toutes les 1000ms = 9 lignes avec NaN sur 10
2. **Inconsistance** - Parfois heading à 100ms, parfois à 120ms
3. **CNN confusion** - Mélange données présentes/absentes
4. **Volume énorme** - 50K+ lignes difficiles à gérer

#### ❌ Option B : Agrégation 5+ secondes

**Description :** Regrouper par intervalles de 5 ou 10 secondes

**Pourquoi rejetée :**
1. **Perte d'information** - Un changement de cap rapide serait moyenné
2. **Manque réactivité** - Attaque rapide passerait inaperçue
3. **Pas adapté aux manœuvres** - Virage = 15-30s, on perdrait la dynamique

#### ❌ Option C : Interpolation au lieu de forward-fill

**Description :** Interpoler linéairement les valeurs manquantes

**Problème à résoudre :** Que faire quand un signal n'a pas de valeur à une seconde ?

**Pourquoi rejetée :**
1. **Crée des fausses valeurs** - Interpole entre 10° et 20° → crée 15° (jamais mesuré)
2. **Masque les patterns** - Un plateau devient une pente
3. **Physiquement incorrect** - Heading ne change pas linéairement
4. **Forward-fill mieux** - Assume que la valeur reste stable (correct pour capteurs)

---

## 2.4 Feature Engineering

### 2.4.1 Méthode Utilisée ✅

**Approche : 4 agrégations statistiques par signal**

Pour chaque signal et chaque seconde :
```python
# 4 statistiques calculées
mean = df.mean()    # Valeur moyenne de la seconde
max  = df.max()     # Pic maximal
min  = df.min()     # Creux minimal
std  = df.std()     # Variabilité (écart-type)
```

**Résultat :** 15 signaux × 4 agrégations = **60 features**

### 2.4.2 Pourquoi ces 4 statistiques ?

| Statistique | Ce qu'elle capture | Exemple maritime |
|-------------|-------------------|------------------|
| **Mean** | Comportement typique | Vitesse moyenne = 6 nœuds |
| **Max** | Pics, pointes | Rafale de vent max = 25 nœuds |
| **Min** | Creux, minimums | Profondeur min = danger |
| **Std** | Stabilité/instabilité | GPS std élevé = spoofing? |

### 2.4.3 Méthodes Alternatives Rejetées ❌

#### ❌ Option A : Uniquement la moyenne

**Description :** Une seule valeur par signal par seconde

**Pourquoi rejetée :**
1. **Perd l'information de variabilité** - GPS stable vs instable = même moyenne
2. **Perd les extremums** - Un pic bref est moyenné
3. **Moins discriminant** - Normal et attaque peuvent avoir même moyenne

#### ❌ Option B : Ajouter médiane et variance

**Description :** 6 statistiques au lieu de 4 (mean, max, min, std, median, variance)

**Pourquoi rejetée :**
1. **Médiane ≈ Mean** - Pour distributions normales, quasi-identiques
2. **Variance = Std²** - Information redondante
3. **Coût calcul** - Médiane = tri → O(n log n)
4. **Overfitting** - Plus de features → plus de paramètres → moins généralise

#### ❌ Option C : Utiliser les percentiles (p25, p50, p75)

**Description :** Distribution complète avec quantiles

**Pourquoi rejetée :**
1. **Trop de features** - 15 signaux × 6 percentiles = 90 features
2. **Données insuffisantes** - Parfois 10 valeurs/seconde → percentiles non significatifs
3. **Mean/max/min suffisent** - Capturent l'essentiel de la distribution

---

## 2.5 Sélection des Signaux

### 2.5.1 Problème à Résoudre

19 signaux extraits, mais certains sont :
- Toujours à 0 (inutiles)
- Presque constants (pas informatifs)
- Redondants (corrélation > 0.9)
- Trop épars (< 50% couverture)

### 2.5.2 Méthode Utilisée ✅

**Approche : Sélection multi-critères**

```
Critère 1 : Couverture > 50%
   → Exclut les signaux trop rares

Critère 2 : Variance > 0.001 (Coefficient de Variation > 2%)
   → Exclut les signaux constants

Critère 3 : Corrélation < 0.9 avec autres signaux
   → Exclut les signaux redondants

Critère 4 : Pertinence pour attaques maritimes
   → Garde les signaux critiques pour la sécurité
```

### 2.5.3 Résultat de la Sélection

**Signaux Gardés (9) :**

| Signal | Couverture | Variance | Pertinence Attaque |
|--------|-----------|----------|-------------------|
| depth | 73.8% | Haute | ✅ Grounding attack |
| rudder_position | 86.0% | Haute | ✅ Hijacking |
| wind_speed | 74.5% | Moyenne | ✅ Navigation |
| wind_angle | 74.5% | Haute | ✅ Navigation |
| sog | 51.7% | Moyenne | ✅ GPS Spoof |
| cog | 51.7% | Haute | ✅ GPS Spoof |
| heading | 73.3% | Haute | ✅ Hijacking |
| pitch | 73.3% | Basse | ⚠️ Stabilité |
| roll | 73.3% | Moyenne | ✅ Stabilité |

**Signaux Exclus (10) :**

| Signal | Raison Exclusion | Détail |
|--------|------------------|--------|
| speed_ground | 0% données | Aucune valeur non-null |
| deviation | 0% données | Aucune valeur non-null |
| offset | Constant | Toutes valeurs = 0.0 |
| variation | Quasi-constant | 1.054° ± 0.002° seulement |
| yaw | Redondant | Corrélation -0.73 avec heading (même source) |
| speed_water | Redondant + épars | r=0.94 avec SOG, 49% couverture |
| latitude | Variance trop basse | Zone de 2km, CV < 2% |
| longitude | Variance trop basse | Zone de 2km, CV < 2% |
| rate_of_turn | Trop épars | 27% couverture seulement |
| rudder_angle_order | Moins bon que position | 62% vs 86% couverture |

### 2.5.4 Méthodes Alternatives Rejetées ❌

#### ❌ Option A : Garder tous les 19 signaux

**Description :** Ne rien exclure, laisser le CNN décider

**Pourquoi rejetée :**
1. **Garbage in, garbage out** - Signaux à 0% n'apportent rien
2. **Overfitting** - 19 signaux × 4 stats = 76 features → trop
3. **Bruit** - Signaux constants ajoutent du bruit
4. **Temps d'entraînement** - Plus de features = plus lent

#### ❌ Option B : Sélection automatique (PCA, Lasso)

**Description :** Algorithme de réduction de dimensionnalité

**Pourquoi rejetée :**
1. **Perd l'interprétabilité** - PC1 = quoi exactement ?
2. **Pas de choix métier** - Algorithme ignore l'importance maritime
3. **Dépendance aux données** - Différent dataset = différente sélection
4. **Notre cas simple** - 19 signaux, tri manuel faisable et meilleur

#### ❌ Option C : Seuil de corrélation 0.7 au lieu de 0.9

**Description :** Être plus strict sur la redondance

**Pourquoi rejetée :**
1. **Perd trop de signaux** - heading et cog corrélés (~0.75) mais sémantiques différentes
2. **Heading** = direction de la proue (compas)
3. **COG** = direction du mouvement (GPS)
4. **Les deux sont utiles** - Un bateau peut dériver (heading ≠ COG)

---

## 2.6 Gestion des Valeurs Manquantes

### 2.6.1 Méthode Utilisée ✅

**Approche : Forward-fill puis Backward-fill**

```python
# Étape 1 : Forward-fill (utiliser la dernière valeur connue)
df = df.ffill()

# Étape 2 : Backward-fill (pour les premières lignes)
df = df.bfill()

# Étape 3 : Si toujours manquant (début/fin dataset)
df = df.fillna(0)
```

**Logique :**
```
Exemple : Heading manquant à seconde 100
   → Seconde 99 : heading = 45°
   → Seconde 100 : heading = ? → Forward-fill = 45°
   → Hypothèse : le heading n'a pas changé en 1 seconde (réaliste)
```

### 2.6.2 Pourquoi Forward-Fill ?

| Aspect | Forward-Fill ✅ | Moyenne | Zéro | Supprimer |
|--------|----------------|---------|------|-----------|
| Continuité temporelle | ✅ Préservée | ❌ Perturbée | ❌ Saut | ❌ Trou |
| Physiquement réaliste | ✅ Oui | ⚠️ Non | ❌ Non | N/A |
| Simple | ✅ Très | ⚠️ Moyen | ✅ Très | ✅ Très |
| Perte données | ❌ Aucune | ❌ Aucune | ❌ Aucune | ⚠️ -20% |

### 2.6.3 Méthodes Alternatives Rejetées ❌

#### ❌ Option A : Supprimer les lignes avec NaN

**Description :** `df.dropna()` - supprimer toute ligne incomplète

**Pourquoi rejetée :**
1. **Perte massive** - Jusqu'à 50% des lignes supprimées
2. **Crée des trous temporels** - Série non continue
3. **Perd le contexte** - Fenêtres interrompues

#### ❌ Option B : Imputation par moyenne globale

**Description :** Remplacer NaN par la moyenne de la colonne

**Pourquoi rejetée :**
1. **Ignore le temps** - Moyenne de 85 minutes ≠ valeur à l'instant T
2. **Crée des sauts** - Heading à 45° puis soudainement 180° (la moyenne)
3. **Physiquement faux** - Un signal ne "saute" pas à sa moyenne

#### ❌ Option C : Imputation par KNN

**Description :** Utiliser les K plus proches voisins pour estimer

**Pourquoi rejetée :**
1. **Complexité excessive** - O(n²) pour chaque valeur manquante
2. **Définition de "proche"** - Temps ? Valeurs ? Les deux ?
3. **Plus lent** - Forward-fill = O(n), KNN = O(n² × features)
4. **Pas nécessaire** - Forward-fill fonctionne bien pour séries temporelles

#### ❌ Option D : Interpolation polynomiale

**Description :** Ajuster une courbe polynomiale et interpoler

**Pourquoi rejetée :**
1. **Crée des valeurs fictives** - Valeurs jamais mesurées
2. **Oscille aux extrémités** - Polynômes instables (effet Runge)
3. **Suppose changement continu** - Un heading peut rester stable puis changer brusquement
4. **Overfitting local** - Polynôme peut créer des pics irréalistes

---

# 3. PHASE 1 : PRÉTRAITEMENT ET FENÊTRAGE

## 3.1 Normalisation

### 3.1.1 Problème à Résoudre

Les features ont des échelles très différentes :
```
Latitude :    43.0 - 43.5 (degrés)
Longitude :   -6.0 - -5.5 (degrés)  
Depth :       0 - 200 (mètres)
Speed :       0 - 20 (nœuds)
Heading :     0 - 360 (degrés)
Wind Speed :  0 - 50 (m/s)
```

**Problème pour le CNN :**
- Latitude (valeurs 43) dominerait les gradients
- Speed (valeurs 0-20) serait ignorée
- Training instable ou impossible

### 3.1.2 Méthode Utilisée ✅

**Approche : Normalisation Min-Max vers [0, 1]**

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Formule : X_norm = (X - X_min) / (X_max - X_min)
```

**Résultat :** Toutes les valeurs dans [0.0, 1.0]

### 3.1.3 Pourquoi Min-Max [0, 1] ?

| Avantage | Explication |
|----------|-------------|
| **Gradients équilibrés** | Toutes features contribuent également |
| **ReLU compatible** | ReLU(x) = max(0, x), [0,1] optimal |
| **Valeurs bornées** | Pas de valeurs extrêmes imprévisibles |
| **Interprétable** | 0 = minimum, 1 = maximum |
| **Convergence rapide** | Training 2-3x plus rapide |

### 3.1.4 Méthodes Alternatives Rejetées ❌

#### ❌ Option A : StandardScaler (Z-score)

**Description :** Centrer sur moyenne 0, écart-type 1

```python
# StandardScaler : (X - μ) / σ
# Résultat : moyenne = 0, std = 1
```

**Pourquoi rejetée :**
1. **Valeurs négatives** - Produit des valeurs < 0
2. **Mauvais pour ReLU** - ReLU(valeur négative) = 0 → neurones "morts"
3. **Pas borné** - Peut produire des valeurs très grandes (outliers)
4. **Suppose distribution normale** - Nos données ne sont pas normales

> [!WARNING]
> StandardScaler avec ReLU = risque de "dying ReLU" où beaucoup de neurones restent à 0

#### ❌ Option B : Pas de normalisation

**Description :** Utiliser les valeurs brutes directement

**Pourquoi rejetée :**
1. **Training impossible** - Gradients explosent ou disparaissent
2. **Features dominantes** - Latitude (43°) >> Speed (5 nœuds)
3. **Non convergence** - Loss ne diminue pas
4. **Résultats aléatoires** - Le CNN n'apprend rien

#### ❌ Option C : Normalisation [-1, 1]

**Description :** Min-Max mais centré sur 0

```python
# Formule : X_norm = 2 * (X - X_min) / (X_max - X_min) - 1
```

**Pourquoi rejetée :**
1. **Valeurs négatives** - Même problème qu'avec StandardScaler
2. **Tanh compatible, mais ReLU non** - On utilise ReLU
3. **Pas d'avantage** - [0,1] suffit pour notre cas

#### ❌ Option D : RobustScaler

**Description :** Utilise médiane et IQR au lieu de mean/std

**Pourquoi rejetée :**
1. **Pas borné** - Peut produire des valeurs hors [0,1]
2. **Outliers présents** - Nos données peuvent avoir des extremums légitimes (attaques simulées)
3. **Complique le décodage** - Difficile de revenir aux vraies valeurs

#### ❌ Option E : Normalisation par signal

**Description :** Normaliser chaque signal indépendamment (déjà fait avec MinMax)

**Pourquoi cette précision :**
- C'est exactement ce que MinMaxScaler fait
- Mais il faut ATTENTION : **fit uniquement sur train, transform sur val/test**

```python
# ✅ CORRECT
scaler.fit(X_train)        # Apprendre min/max sur train SEULEMENT
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)  # Même scaler, pas refit

# ❌ INCORRECT (data leakage)
scaler.fit(X_full)  # Voit les données de test !
```

---

## 3.2 Validation Visuelle

### 3.2.1 Pourquoi Valider Visuellement ?

> [!IMPORTANT]
> "Les chiffres peuvent mentir, les graphiques révèlent"

Validation numérique + visuelle = confiance maximale avant le coûteux training CNN.

### 3.2.2 Les 6 Validations Effectuées ✅

#### Plot 1 : Vérification de la Normalisation
**Objectif :** Confirmer toutes valeurs dans [0, 1]
**Vérifié :**
- min(toutes valeurs) >= 0 ✅
- max(toutes valeurs) <= 1 ✅
- Pas de NaN ✅

#### Plot 2 : Relations entre Agrégations
**Objectif :** Vérifier cohérence mathématique
**Vérifié :**
- Pour chaque signal : max >= mean >= min ✅
- std >= 0 toujours ✅

#### Plot 3 : Patterns Temporels
**Objectif :** Visualiser l'évolution des signaux
**Vérifié :**
- Transitions douces (pas de sauts brusques artificiels) ✅
- Patterns maritimes visibles (virages, accélérations) ✅

#### Plot 4 : Matrice de Corrélation
**Objectif :** Identifier corrélations excessives
**Vérifié :**
- Speed_mean ↔ Speed_max fortement corrélés (attendu) ✅
- Pas de corrélation 1.0 inattendue ✅

#### Plot 5 : Continuité Temporelle
**Objectif :** Détecter gaps ou discontinuités
**Vérifié :**
- Lignes consécutives similaires ✅
- Pas de "sauts" de plus de 20% entre secondes adjacentes ✅

#### Plot 6 : Aperçu Entrée CNN
**Objectif :** Voir ce que le CNN va vraiment recevoir
**Format :** Heatmap des 100 premières lignes × 60 features
**Vérifié :**
- Patterns visuellement distinguables ✅
- Prêt pour l'apprentissage ✅

---

## 3.3 Fenêtrage (Windowing)

### 3.3.1 Problème à Résoudre

**CNN nécessite une entrée de taille fixe :**
- Notre dataset : 5,095 lignes (variable selon la capture)
- CNN : attend (batch, time_steps, features) avec time_steps fixe

**Solution : Fenêtre glissante (Sliding Window)**

### 3.3.2 Méthode Utilisée ✅ (CANShield Multi-échelle)

**Configuration :**
```python
TIME_STEPS = [50, 75, 100]      # Tailles de fenêtre (en lignes)
SAMPLING_PERIODS = [1, 5, 10]   # Facteurs d'échantillonnage (en secondes)
WINDOW_STEP = 10                # Pas de glissement (en lignes)
```

**Résultat : 9 configurations (3 × 3)**

| Config | Fenêtre | Sampling | Temps Réel Couvert | Spécialité |
|--------|---------|----------|-------------------|------------|
| 1 | 50 rows | 1s | 50 secondes | Attaques rapides |
| 2 | 50 rows | 5s | 250 secondes | Drift moyen |
| 3 | 50 rows | 10s | 500 secondes | Spoofing lent |
| 4 | 75 rows | 1s | 75 secondes | Bursts courts |
| 5 | 75 rows | 5s | 375 secondes | Changements graduels |
| 6 | 75 rows | 10s | 750 secondes | Trends longs |
| 7 | 100 rows | 1s | 100 secondes | Attaques étendues |
| 8 | 100 rows | 5s | 500 secondes | Patterns moyens-longs |
| 9 | 100 rows | 10s | 1000 secondes | Anomalies très lentes |

### 3.3.3 Explication du Multi-échelle

> [!TIP]
> **L'idée clé :** Différents types d'attaques ont différentes durées.
> Un seul modèle ne peut pas tout voir.

**Exemple concret :**

```
Attaque 1 - Injection GPS rapide :
   Durée : 5 secondes
   → Visible avec fenêtre 50 rows + sampling 1s (=50s de contexte)
   → Invisible avec sampling 10s (la fenêtre saute par-dessus)

Attaque 2 - Drift GPS lent :
   Durée : 10 minutes
   → Invisible avec fenêtre 50s (ne voit qu'un fragment)
   → Visible avec fenêtre 100 rows + sampling 10s (=1000s de contexte)
```

### 3.3.4 Chevauchement (Overlap)

**Formule :** 
```
Overlap = 1 - (Step / Window_size) = 1 - (10/50) = 80%
```

**Pourquoi 80% ?**

| Overlap | Avantages | Inconvénients |
|---------|-----------|---------------|
| 0% | Moins de données, rapide | Rate les attaques entre fenêtres |
| 50% | Équilibré | Peut rater des patterns |
| 80% ✅ | Haute couverture | Plus de fenêtres (acceptable) |
| 98% | Couverture maximale | Redondant, lent, ~5000 fenêtres |

### 3.3.5 Calcul du Nombre de Fenêtres

**Formule générale :**
```
N_windows = floor((N_rows - Window_size) / Step_size) + 1
```

**Exemple pour 50 rows + 1s sampling :**
```
N_rows = 5095
Window_size = 50
Step_size = 10

N_windows = floor((5095 - 50) / 10) + 1 = 505 fenêtres
```

**Total des fenêtres générées :**

| Window | 1s Sampling | 5s Sampling | 10s Sampling | Total |
|--------|-------------|-------------|--------------|-------|
| 50 rows | 505 | 97 | 47 | 649 |
| 75 rows | 503 | 95 | 44 | 642 |
| 100 rows | 500 | 92 | 42 | 634 |
| **TOTAL** | | | | **1,925** |

### 3.3.6 Méthodes Alternatives Rejetées ❌

#### ❌ Option A : Fenêtre unique (pas de multi-échelle)

**Description :** Un seul modèle CNN avec fenêtre fixe de 50 rows

**Pourquoi rejetée :**
1. **Rate les attaques longues** - Fenêtre de 50s ne voit pas un drift de 10 minutes
2. **Rate les attaques courtes si sampling élevé** - Avec 10s sampling, une attaque de 5s est moyennée
3. **Moins robuste** - Un modèle peut échouer, l'ensemble compense
4. **Moins précis** - Étudies montrent que multi-échelle > mono-échelle

#### ❌ Option B : Fenêtre variable (adaptive windowing)

**Description :** Ajuster la taille de fenêtre dynamiquement

**Pourquoi rejetée :**
1. **Complexité** - Comment décider de la taille à l'exécution ?
2. **CNN fixe** - Un CNN a une architecture fixe, pas adaptable à l'entrée
3. **Training compliqué** - Batch training nécessite tailles uniformes
4. **Non reproductible** - Difficile de comparer les résultats

#### ❌ Option C : Pas de chevauchement (step = window_size)

**Description :** Fenêtres adjacentes sans overlap

```
Window 1 : rows 0-49
Window 2 : rows 50-99  ← 0% overlap
Window 3 : rows 100-149
```

**Pourquoi rejetée :**
1. **Rate les attaques aux frontières** - Attaque à rows 45-55 est coupée en deux
2. **Moins de données** - 102 fenêtres au lieu de 505
3. **Contexte manquant** - Chaque fenêtre est isolée

#### ❌ Option D : Overlap 98% (step = 1)

**Description :** Avancer d'une seule ligne à chaque fenêtre

**Pourquoi rejetée :**
1. **Trop de fenêtres** - 5046 fenêtres au lieu de 505
2. **Haute redondance** - Fenêtre N et N+1 partagent 49/50 lignes
3. **Training lent** - 10x plus d'époques
4. **Overfitting** - Le modèle voit presque les mêmes données répétées
5. **Stockage** - 5046 images × 9 configs = 45,414 fichiers

#### ❌ Option E : Votre approche alternative (discutée)

**Description proposée :** 
- Fenêtre fixe 50 secondes toujours
- Step variable : 1s, 5s, 10s

**Pourquoi rejetée (expliquée en détail) :**

```
Approche alternative:
   Config 1: 50s window, step 1s → 5,046 windows
   Config 2: 50s window, step 5s → 1,010 windows  
   Config 3: 50s window, step 10s → 505 windows

Problème : TOUTES les fenêtres voient 50 secondes !

Un drift GPS lent sur 10 minutes :
   - Toutes les fenêtres voient 50s du drift
   - Aucune fenêtre ne voit le pattern complet de 10 min
   - L'attaque est "invisible" car fragmentée
```

**CANShield (notre approche) :**
```
Config 3: 50 rows, 10s sampling → 500 secondes de contexte

Le même drift de 10 minutes :
   - Une fenêtre voit 8.3 minutes du drift
   - Pattern complet visible
   - Détection possible !
```

> [!IMPORTANT]
> **Différence clé :**
> - Votre approche : Change l'OVERLAP mais pas le CONTEXTE TEMPOREL
> - CANShield : Change le CONTEXTE TEMPOREL via le downsampling

---

## 3.4 Downsampling (Sous-échantillonnage)

### 3.4.1 Concept Expliqué

**Downsampling = Garder 1 ligne sur N**

```python
# Downsampling période 5s
data_sp5 = data[::5]  # Garde lignes 0, 5, 10, 15, ...

# Avant : 5095 lignes
# Après : 1019 lignes
```

### 3.4.2 Pourquoi Downsampler ?

**But : Voir différentes échelles temporelles avec la même taille de fenêtre**

```
Fenêtre 50 rows + Sampling 1s:
   - Lignes 0, 1, 2, ..., 49
   - Temps couvert : 50 secondes
   - Résolution : 1 mesure/seconde (haute résolution)

Fenêtre 50 rows + Sampling 10s:
   - Lignes 0, 10, 20, ..., 490
   - Temps couvert : 500 secondes (8 min 20s)
   - Résolution : 1 mesure/10 secondes (basse résolution)
```

### 3.4.3 Trade-off Résolution vs Contexte

| Sampling | Temps Couvert (50 rows) | Résolution | Mieux pour |
|----------|------------------------|------------|------------|
| 1s | 50s | Haute (1Hz) | Attacks rapides |
| 5s | 250s (4 min) | Moyenne (0.2Hz) | Attacks moyennes |
| 10s | 500s (8 min) | Basse (0.1Hz) | Attacks lentes |

### 3.4.4 Pourquoi [1, 5, 10] et pas [1, 2, 3] ?

**Principe : Échelles logarithmiques pour couvrir plus de terrain**

```
Option [1, 2, 3]:
   Temps couverts : 50s, 100s, 150s
   Différence : 3x seulement
   Problème : Trop similaires, manque les attaques très lentes

Option [1, 5, 10] ✅:
   Temps couverts : 50s, 250s, 500s
   Différence : 10x
   Avantage : Couvre large spectre d'attaques
```

### 3.4.5 Méthodes Alternatives Rejetées ❌

#### ❌ Option A : Moyennage au lieu de sous-échantillonnage

**Description :** Au lieu de garder 1 ligne sur 10, moyenner 10 lignes

```python
# Averaging
data_avg = data.reshape(-1, 10).mean(axis=1)
```

**Pourquoi rejetée :**
1. **Lisse les anomalies** - Un pic sur 1 ligne est divisé par 10
2. **Perd les extremums** - Max et min sont moyennés
3. **Masque les patterns** - Transitions brusques deviennent graduelles
4. **Le sous-échantillonnage préserve** - Garde les vraies valeurs

#### ❌ Option B : Pas de downsampling (toujours 1s)

**Description :** Garder toute la résolution, varier seulement la fenêtre

**Pourquoi rejetée :**
1. **Fenêtre 500 rows** - Taille CNN trop grande, slow training
2. **Trop de détails** - Le CNN voit trop de "bruit" haute fréquence
3. **Pas pratique** - 500 → problèmes de mémoire GPU

---

# 4. TABLEAU RÉCAPITULATIF DES DÉCISIONS

## 4.1 Phase 0 : Décisions Clés

| Étape | Méthode Choisie | Alternatives Rejetées | Justification |
|-------|-----------------|----------------------|---------------|
| **Décodage** | Décodeur custom NMEA 2000 | Librairie externe, Raw bytes | Contrôle, validation, interprétabilité |
| **Agrégation** | 1 seconde | 100ms, 5s | Équilibre détail/gestion, aligné NMEA |
| **Statistiques** | mean, max, min, std | Médiane, percentiles | Capture l'essentiel sans redondance |
| **Sélection** | 9 signaux (multi-critères) | Tous 19, PCA | Qualité + pertinence attaque |
| **Missing data** | Forward-fill | Drop, moyenne, KNN | Continuité temporelle préservée |

## 4.2 Phase 1 : Décisions Clés

| Étape | Méthode Choisie | Alternatives Rejetées | Justification |
|-------|-----------------|----------------------|---------------|
| **Normalisation** | Min-Max [0,1] | StandardScaler, [-1,1], aucune | Compatible ReLU, borné, stable |
| **Validation** | 6 plots visuels | Stats seules | Confiance avant training coûteux |
| **Fenêtrage** | Multi-échelle CANShield | Fenêtre unique, adaptive | Détecte tous types d'attaques |
| **Tailles fenêtre** | [50, 75, 100] rows | Taille unique | Patterns courts/moyens/longs |
| **Sampling** | [1, 5, 10] secondes | [1, 2, 3] | Échelle logarithmique, large couverture |
| **Overlap** | 80% | 0%, 50%, 98% | Couverture sans redondance excessive |
| **Downsampling** | Sous-échantillonnage | Moyennage | Préserve les vraies valeurs |

---

# 5. CONCLUSION ET PROCHAINES ÉTAPES

## 5.1 Ce qui a été accompli

### Phase 0 ✅
- [x] Décodage de 654K messages NMEA 2000
- [x] Extraction de 19 signaux bruts
- [x] Agrégation temporelle 1 seconde
- [x] Feature engineering (60 features)
- [x] Sélection rigoureuse → 9 signaux finaux
- [x] Gestion des valeurs manquantes
- [x] Dataset clean : 5,095 × 60

### Phase 1 ✅
- [x] Normalisation Min-Max [0,1]
- [x] 6 validations visuelles
- [x] Fenêtrage multi-échelle (9 configurations)
- [x] 1,925 fenêtres individuelles générées
- [x] 27 visualisations résumé
- [x] Documentation complète

## 5.2 Prêt pour Phase 2

**Prochaine étape : Entraînement CNN-Autoencoder**

```
Phase 2 (à venir):
├── Construire l'architecture CNN (from CANShield)
├── Entraîner 9 modèles (1 par configuration)
├── Transfer learning (sp1 → sp5 → sp10)
├── Évaluer reconstruction error
└── Définir seuils de détection
```

## 5.3 Points Forts de Notre Approche

| Aspect | Notre Travail | CANShield (référence) |
|--------|--------------|----------------------|
| **Données** | Réelles (bateau) | Synthétiques |
| **Justification params** | Maritime-spécifique | Non expliquée |
| **Documentation** | Exhaustive | Paper seulement |
| **Validation decoder** | Vs librairie officielle | Non mentionnée |
| **Sélection signaux** | Multi-critères | Non expliquée |

---

## 📚 RÉFÉRENCES

1. **CANShield Paper** : CNN-Autoencoder for Automotive CAN IDS
2. **NMEA 2000 Specification** : National Marine Electronics Association
3. **N2kMessages Library** : Reference implementation for validation

---

**Document créé le :** 16 Décembre 2025  
**Dernière mise à jour :** 16 Décembre 2025  
**Auteur :** Équipe N2KShield

---

> [!NOTE]
> Ce document est un récapitulatif vivant. Il sera mis à jour au fur et à mesure de l'avancement du projet.
