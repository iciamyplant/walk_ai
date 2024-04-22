# walk_ai




## 1.  Reinforcement Learning
#### a. Définition

Branche de l'intelligence artificielle où un agent apprend à prendre des décisions en interagissant avec son environnement. C'est à dire que l'agent prend des actions dans un environnement et reçoit des récompenses ou des pénalités en fonction de ces actions, sachant que l’objectif est de trouver un modèle d’action approprié qui maximiserait la récompense cumulée totale de l’agent. ⇒ il doit donc apprendre quelle action prendre dans quelle situation pour maximiser sa récompense globale. Contrairement à d'autres formes d'apprentissage supervisé ou non supervisé, le renforcement ne nécessite pas d'exemples étiquetés de données en entrée ou de résultats souhaités.

<p align="center">
<img width="700" src="https://github.com/iciamyplant/walk_ai/assets/57531966/a101cebd-16b7-4251-9d3e-94a4c9a0a694">
<p align="center">

Concept
- un agent, représente celui qui apprend
- takes actions at each of time t : At
- the action is received by the environnement
- qui l'utilise pour produire une récompense Rt+1, et un état à ce moment du temps St+1
- ceci est renvoyé à l'agent pour que la prochaine action soit décidée
- le processus se repète

time is discrete : t=[0,1,2,3,...]
souvent les états, les actions et les récompenses sont définis, et l'agent apprend à travers l'exploration et l'exploitation comment agir de manière optimale dans ces états pour maximiser les récompenses


#### b. Cas d'utilisation

Adaptés dans les situations où il faut prendre des décisions les unes après les autres dans un environnement incertain pour atteindre des objectifs spécifiques. Applications dans de nombreux domaines :
- Les jeux : Deepmind impressive version of RL : Alpha zero devenant l'un des meilleurs moteurs d'échecs utilisant seulement quelques heures de jeu, Muzero atteignant un niveau de performance similaire mais sur un esemble de taches bcp plus diversifié : atari games, go. Mais aussi AlphaGo de Deepmind. PrefixRL, Nvidia a annoncé avoir utilisé du deep RL pour concevoir des circuits airthmétiques plus approfondis. Le RL a été largement utilisé pour créer des agents autonomes capables de jouer à des jeux vidéo comme Dota 2, StarCraft II, ou des jeux de plateau comme Go, échecs et dame
- Robotique : Les robots autonomes peuvent apprendre à manipuler des objets, à marcher, à naviguer dans des environnements complexes et à accomplir diverses tâches en utilisant le RL pour s'adapter à des conditions changeantes.
- Publicité en ligne : Les algorithmes de RL sont souvent utilisés pour optimiser les campagnes publicitaires en ligne, en ajustant dynamiquement les enchères et les stratégies de diffusion pour maximiser le retour sur investissement. plateforme utilise le RL pour ajuster dynamiquement les enchères et les stratégies de diffusion des annonces afin d'optimiser les performances de la campagne publicitaire. Les récompenses pourraient être basées sur des métriques telles que le nombre de clics, le taux de conversion, le chiffre d'affaires généré, etc. Au fur et à mesure que de nouvelles données sont disponibles, l'agent ajuste ses décisions en temps réel pour s'adapter aux changements dans l'environnement, tels que les tendances du marché, la concurrence, ou les préférences des utilisateurs.
- Contrôle de la circulation et logistique : Le RL peut être appliqué pour optimiser la circulation routière, la gestion des feux de signalisation, la logistique de livraison et la planification des itinéraires
- La gestion des ressources, la finance...

#### c. Algos 

Le RL fait appel à une variété d'algorithmes et de techniques informatiques pour permettre à un agent d'apprendre à prendre des décisions dans un environnement donné, quelques unes de ces méthodes :

|Method|Definition|
|-----|-----|
|Q-Learning|C'est l'un des algorithmes les plus fondamentaux du RL. Il est utilisé pour apprendre une fonction d'évaluation de l'action appelée fonction Q, qui indique la valeur d'une action dans un état donné|
|SARSA (State-Action-Reward-State-Action)|C'est un autre algorithme basé sur la programmation dynamique qui est utilisé pour apprendre une politique d'action optimale dans un environnement de RL|













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






