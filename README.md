# walk_ai

## 1.  Reinforcement Learning and Q Learning

Deepmind impressive version of RL : Alpha zero devenant l'un des meilleurs moteurs d'échecs utilisant seulement quelques heures de jeu, Muzero atteignant un niveau de performance similaire mais sur un esemble de taches bcp plus diversifié : atari games, go.
PrefixRL, Nvidia a annoncé avoir utilisé du deep RL pour concevoir des circuits airthmétiques plus approfondis.
- the bellman equations, dynamic programming, genereliazed policy iteration
- monte carlo and off policy methods
- td learning, sarsa, and q learning
- function approximation
- policy gradient methods

Concept
- un agent, représente celui qui apprend
- takes actions at each of time t : At
- the action is received by the environnement
- qui l'utilise pour produire une récompense Rt+1, et un état à ce moment du temps St+1
- ceci est renvoyé à l'agent pour que la prochaine action soit décidée
- le processus se repète

time is discrete : t=[0,1,2,3,...]

reinforcement learning
q learning
Snake Game Example

## 2. Humanoide 2D processus d'optimisation ARS

Augmented Random Search Algorithm
- ARS is a latest A.I algorithm proposed in march 2018 dans cet article : [paper](https://arxiv.org/pdf/1803.07055.pdf)
- As in every RL problem, the target is to find a policy to maximize the expected reward that the agent might get when following this policy in the given environment (= trouver une politique permettant de maximiser la récompense attendue que l'agent pourrait obtenir en suivant cette politique dans l'environnement donné)
- It almost 15 times faster than other reinforcement algorithms of 2017.
- ARS is based on shallow learning method
- ARS exemple 2D gym humanoide [ex](https://www.youtube.com/watch?v=TVsPttCWeOo)

La solution proposée par l'article consiste à améliorer un algorithme existant appelé Basic Random Search. Basic random search = L'idée de la recherche aléatoire de base est de choisir une politique pramatérisée 𝜋𝜃, de choquer (ou de perturber) les paramètres 𝜃 en appliquant +𝛎𝜹 et -𝛎𝜹 (où 𝛎 < 1 est un bruit constant et 𝜹 est un nombre aléatoire généré à partir d'une distribution normale). Appliquez ensuite les actions basées sur 𝜋(𝜃+𝛎𝜹) et 𝜋(𝜃-𝛎𝜹) puis récupérez les récompenses r(𝜃+𝛎𝜹) et r(𝜃-𝛎𝜹) résultant de ces actions. Maintenant que nous avons les récompenses du 𝜃 perturbé, calculez la moyenne Δ = 1/N * Σ[r(𝜃+𝛎𝜹) - r(𝜃-𝛎𝜹)]𝜹 pour tous les 𝜹 et on met à jour les paramètres 𝜃 en utilisant Δ et un taux d'apprentissage 𝝰.𝜃ʲ⁺¹ = 𝜃ʲ + 𝝰.Δ



Il possède une couche d'entrée qui accepte un vecteur de l'état de l'environnement
puis, après les avoir multipliés par les poids et passé par une fonction d'activation
il donne une sortie à l'agent sur l'action qu'il doit effectuer sur l'environnement

Il est différent des autres modèles d'apprentissage par renforcement car il n'a pas de réseau neuronal profond entre ses couches d'entrée et de sortie. De plus, il ne fonctionne pas sur l'espace d'action, il fonctionne sur l'espace politique.

ARS utilise une approche différente pour optimiser son poids. Contrairement à d’autres algorithmes, il utilise une approche par différences finies pour optimiser les poids. f'(x) = summation(f(a+h)-f(a)) où h est une très petite pertubation.


## 3. Humanoide 3D avec processus PPOT







-------
Monte carlo method



Intall VM
- ajouter Vbox Guest additions dans stockage
- config > general > avancé > presse-papier bidirectionnel
- sudo apt install linux-headers-$(uname -r) build-essential dkms
- redemarrer la vm
- [tuto](https://www.youtube.com/watch?v=MI1THQJFZXY)






