# --------------------------------------------------
# Projet Métaheuristiques
# DUMOULIN Simon, JAMIN Antoine, PAGNON Alexis, SANCHEZ Adam
# 09/09/2025
# --------------------------------------------------

import docplex
import docplex.mp
import docplex.mp.model

# Ouvrir le fichier avec nos variables (code récupéré du projet de S8)
airland_file = open("airlands/airland2.txt")
m = 2  # Nombre de pistes d'atterrissage

# Représenter la solution
solution =  [ [] for _ in range(m) ] # Liste de m listes (1 liste par piste) avec l'ordre des avions qui y atterissent

# Charger les données
"""
The format of these data files is:
number of planes (p), freeze time
for each plane i (i=1,...,p):
   appearance time, earliest landing time, target landing time,
   latest landing time, penalty cost per unit of time for landing
   before target, penalty cost per unit of time for landing
   after target
   for each plane j (j=1,...p): separation time required after i lands before j can land
"""
E = []  # Tableau des Ei (earliest landing time)
T = []  # Tableau des Ti (target landing time)
L = []  # Tableau des Li (latest landing time)
s = []  # Tableau des s_i, étant eux-même des tableaux de s_ij
c_minus = []  # Coût de pénalité par unité de temps pour atterrissage avant l'heure cible
c_plus = []  # Coût de pénalité par unité de temps pour atterrissage après l'heure cible

line = airland_file.readline().strip()
n = int(line.split(" ")[0]) # Nombre d'avions

# Lire le fichier ligne par ligne
for i in range(0, n):
    line = airland_file.readline().strip()

    # Ajouter les valeurs dans les listes appropriées
    values = line.split(" ")

    E.append(float(values[1]))
    T.append(float(values[2]))
    L.append(float(values[3]))

    c_minus.append(float(values[4]))  # Coût de pénalité pour atterrissage prématuré
    c_plus.append(float(values[5]))  # Coût de pénalité pour atterrissage tardif

    # Récupère les S_i
    s_i = []
    # Puisque les valeurs de S sont réparties sur plusieurs lignes, on utilise un while sur la longueur de
    # notre tableau pour les récupérer sans lire une ligne en trop
    while (len(s_i) < n):
        line = airland_file.readline().strip()  # On lit une ligne
        values = line.split(" ")
        for v in values:  # On ajoute toutes les valeurs d'une ligne à s_i, et si la taille de s_i n'est pas égal à n, on lit la ligne suivante
            s_i.append(float(v))
    s.append(s_i)  # On ajoute les valeurs de s_i de cet avion à s qui comprend les valeurs s_i de tous les avions

airland_file.close()


#### AFFICHAGE DES RÉSULTATS #####
print(f"Nombre d'avions: {n}")
# print(f"Heures d'atterrissage au plus tôt: {E}")
# print(f"Heures d'atterrissage cibles: {T}")
# print(f"Heures d'atterrissage au plus tard: {L}")
# print(f"Coûts de pénalité pour atterrissage avant la cible: {c_minus}")
# print(f"Coûts de pénalité pour atterrissage après la cible: {c_plus}")
# print(f"Exemple de temps de séparation (premier avion): {s[0]}")

def genererSequence() :
    return 0

def decode(sequence): 
    """
    Décodage d'une séquence d'atterrissage (on vérifie que la solution est possible et on calcule son coût)
    Retourne :
        x : dictionnaire des temps d'atterrissage
        cost : coût total de la séquence (infini si infaisable)
        feasible : booléen indiquant si la séquence est faisable
    """
    x = {}
    cost = 0
    for r in sequence:
        previous = None
        for i in r:
            
            if previous is None:
                t = max(E[i], 0)
            else:
                t = max(E[i], x[previous] + s[previous][i])
            if t > L[i]:  # infaisable
                return x, float("inf"), False
            x[i] = t
            # coût = avance ou retard
            if t < T[i]:
                cost += c_minus[i] * (T[i] - t)
            else:
                cost += c_plus[i] * (t - T[i])
            previous = i
    return x, cost, True

def decode2(sequence):
    """
    Décodage d'une séquence d'atterrissage (on vérifie que la solution est possible et on calcule son coût)
    Retourne :
        x : dictionnaire des temps d'atterrissage
        cost : coût total de la séquence (infini si infaisable)
        feasible : booléen indiquant si la séquence est faisable
    """
    x = {}
    cost = 0
    for r in sequence:
        previous = None
        for i in r:
            # temps le plus tôt possible selon les contraintes
            t_min = E[i] if previous is None else max(E[i], x[previous] + s[previous][i])
            t_max = L[i]

            # choisir le temps le plus proche de T[i] dans [t_min, t_max]
            if T[i] < t_min:
                t = t_min
            elif T[i] > t_max:
                t = t_max
            else:
                t = T[i]

            x[i] = t

            # coût = avance ou retard
            if t < T[i]:
                cost += c_minus[i] * (T[i] - t)
            else:
                cost += c_plus[i] * (t - T[i])

            previous = i

    return x, cost, True

def decode_docplex(sequence):
    """
    Décode une séquence d'atterrissage en trouvant les temps optimaux
    minimisant le coût avec Docplex.
    Procedure to verify if a given solution is feasible or not
    """
    mdl = docplex.mp.model.Model("TempsSequence")
    
    # variables pour les temps d'atterrissage
    x = {i: mdl.continuous_var(lb=E[i], ub=L[i], name=f"x_{i}") for r in sequence for i in r}
    
    # variables pour avance et retard
    early = {i: mdl.continuous_var(lb=0, name=f"early_{i}") for r in sequence for i in r}
    late  = {i: mdl.continuous_var(lb=0, name=f"late_{i}") for r in sequence for i in r}
    
    # contraintes d'avance/retard
    for r in sequence:
        for i in r:
            mdl.add_constraint(T[i] - x[i] == early[i] - late[i])
    
    # contraintes de précédence
    for r in sequence:
        for idx in range(1, len(r)):
            prev = r[idx-1]
            curr = r[idx]
            mdl.add_constraint(x[curr] >= x[prev] + s[prev][curr])
    
    # fonction objectif : minimiser le coût
    mdl.minimize(mdl.sum(c_minus[i]*early[i] + c_plus[i]*late[i] for r in sequence for i in r))
    
    # résoudre
    solution = mdl.solve(log_output=False)
    if solution is None:
        return {}, float("inf"), False
    
    # récupérer les temps optimaux
    x_opt = {i: x[i].solution_value for r in sequence for i in r}
    cost = mdl.objective_value
    
    return x_opt, cost, True




# print(decode_docplex([[2,3,4,7,6,5,8,9,13,12,0,1,11,10,14]]))

# print(decode_docplex([[2,4,7,8,0,13,12,1,11,10],[3,5,6,9,14]]))



def create_initial_solution() :
    sequence = [ [] for _ in range(m)]
    liste_avions = list(range(n))
    liste_avions.sort(key=lambda x: T[x])

    avions_restants = liste_avions.copy()
    for r in range(m):
        previous = None
        for avion in avions_restants[:] :
            if previous is None :
                sequence[r].append(avion)
                previous = avion
                avions_restants.remove(avion)
            else:
                t = max(T[avion], T[previous]+s[previous][avion])
                if t > L[avion]:
                    continue
                else:
                    sequence[r].append(avion)
                    previous = avion
                    avions_restants.remove(avion)
    return sequence

# solution_initiale = create_initial_solution()
# print("Solution initiale : ", solution_initiale)
# print(decode_docplex(solution_initiale))


# TODO : Générer voisins -> Local Search (swap, relocate, move_between_runways, stop criteria), VNS, Tabu Search