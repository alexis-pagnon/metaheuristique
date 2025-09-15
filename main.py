# --------------------------------------------------
# Projet Métaheuristiques
# DUMOULIN Simon, JAMIN Antoine, PAGNON Alexis, SANCHEZ Adam
# 09/09/2025
# --------------------------------------------------

import time
import docplex
import docplex.mp
import docplex.mp.model
import random
import copy

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
    
    # Variables pour les temps d'atterrissage
    x = {i: mdl.continuous_var(lb=E[i], ub=L[i], name=f"x_{i}") for r in sequence for i in r}
    
    # Variables pour avance et retard
    early = {i: mdl.continuous_var(lb=0, name=f"early_{i}") for r in sequence for i in r}
    late  = {i: mdl.continuous_var(lb=0, name=f"late_{i}") for r in sequence for i in r}
    
    for r in sequence:
        # Contraintes d'avance/retard
        for i in r:
            mdl.add_constraint(T[i] - x[i] == early[i] - late[i])

        # Contraintes de précédence
        for idx in range(1, len(r)):
            prev = r[idx-1]
            curr = r[idx]
            mdl.add_constraint(x[curr] >= x[prev] + s[prev][curr])
    
    # Fonction objectif : minimiser le coût
    mdl.minimize(mdl.sum(c_minus[i]*early[i] + c_plus[i]*late[i] for r in sequence for i in r))
    
    # Résolution
    solution = mdl.solve(log_output=False)
    if solution is None:
        return {}, float("inf"), False
    
    # Récupérer les temps optimaux
    x_opt = {i: x[i].solution_value for r in sequence for i in r}
    cost = mdl.objective_value
    
    return x_opt, cost, True




# print(decode_docplex([[2,3,4,7,6,5,8,9,13,12,0,1,11,10,14]]))

# print(decode_docplex([[2,4,7,8,0,13,12,1,11,10],[3,5,6,9,14]]))

# print(decode_docplex([[0, 7, 3, 9, 8, 10, 2, 19, 1, 6, 14, 4, 23, 17, 13, 22, 16, 49, 25, 24, 42, 15, 21, 43, 48, 27, 31, 28, 46, 33, 37, 20, 38, 45, 35, 39, 41], [5, 11, 18, 12, 34, 26, 44, 32, 36, 47, 29, 30, 40]]))

def create_initial_solution() :
    """ Créer une solution initiale non optimisée (On prend la première qui est faisable) """
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
# Possible d'implémenter l'approximation pour comparer avec docplex le résultat final

def relocate_intra(seq, r, i, j)->list:
    """ 
    Retire l'avion d'indice 'i' de la piste 'r' et le réinsère sur celle-ci à l'indice 'j'.
    Renvoie la nouvelle séquence trouvée.
    """
    new_seq = copy.deepcopy(seq)

    if len(new_seq[r]) < 2:
        return new_seq
    
    # On enlève l'avion i 
    plane = new_seq[r].pop(i)
    
    # Puis on le replace à l'emplacement j
    new_seq[r].insert(j, plane)

    return new_seq


def move_between(seq, r1, r2, i1, i2)->list:
    """ 
    Déplace l'avion d'indice i1 de la piste r1 vers la piste r2, à l'indice i2 
    Renvoie la nouvelle séquence trouvée.
    """
    new_seq = copy.deepcopy(seq)

    if len(new_seq) < 2:
        return new_seq
    
    if not new_seq[r1]:
        return new_seq
    
    # On enlève l'avion i1 sur la première piste r1
    plane = new_seq[r1].pop(i1)

    # Puis on le replace à l'emplacement i2 sur la seconde piste r2
    new_seq[r2].insert(i2, plane)

    return new_seq


def LS1(seq, t_debut, t_max):
    """ 
    Algorithme de Local Search permettant à partir d'une séquence donnée de nous renvoyer la meilleure solution (séquence et coût) 
    trouvée dans un temps imparti. Elle crée le voisinage en faisant toutes les combinaisaisons possibles pour chacunes des pistes indépendamment.
    """
                
    best_seq = seq
    _, initial_cost, _ = decode_docplex(best_seq)
    best_cost = initial_cost

    # Condition d'arrêt temporelle
    if(t_debut + t_max < time.time()):
        return best_seq, best_cost
    
    # On crée notre voisinage 
    neighborhoods = []
    for r in range(m):
        for i in range(len(seq[r])):
            for j in range(i+1, len(seq[r])):
                neighborhoods.append(relocate_intra(seq, r, i, j))

    # On calcul le coût de chacunes des séquences trouvées afin de garder la meilleure
    for voisin in neighborhoods:
        _, new_cost, new_feasible = decode_docplex(voisin)
        if new_feasible and new_cost < best_cost:
            best_seq, best_cost = voisin, new_cost

    if(best_cost < initial_cost):
        return LS1(best_seq, t_debut, t_max)
    else:
        return best_seq, best_cost


# print(LS1([[2,7,8,0,13,4,11,1,12,10],[3,5,6,9,14]], time.time(), 10))






def LS2(seq, max_iter = 100):
    """ 
    Algorithme de Local Search permettant à partir d'une séquence donnée de nous renvoyer la meilleure solution trouvée 
    lorsque l'on atteint le nombre maximum d'itérations donné. Elle effectue à chaque fois les 3 types de voisnages différents.
    """
    best_seq = seq
    _, best_cost, feasible = decode_docplex(best_seq)

    # On regarde dans un premier temps si c'est faisable
    if not feasible:
        return best_seq, float("inf")
    
    # On cherche une solution optimale tant que nous n'avons pas atteint le max d'itérations souhaité


    return best_seq, best_cost



# Dans rapport, parler de l'approximation dans les limites.