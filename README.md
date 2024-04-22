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

<p align="center">
<img width="400" src="https://github.com/iciamyplant/walk_ai/assets/57531966/833def76-8757-4140-a907-4e5eb55c2229">
<p align="center">

**Markov Decision Processes(MDPs)** = are mathematical frameworks to describe an environment in RL and almost all RL problems can be formulated using MDPs. An MDP consists of a set of finite environment states S, a set of possible actions A(s) in each state, a real valued reward function R(s) and a transition model P(s’, s | a). However, real world environments are more likely to lack any prior knowledge of environment dynamics. Model-free RL methods come handy in such cases.

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

|Method|Definition|Comments
|-----|-----|-----|
|Q-Learning|C'est l'un des algorithmes les plus fondamentaux du RL. Il est utilisé pour apprendre une fonction d'évaluation de l'action appelée fonction Q, qui indique la valeur d'une action dans un état donné||
|SARSA (State-Action-Reward-State-Action)|C'est un autre algorithme basé sur la programmation dynamique qui est utilisé pour apprendre une politique d'action optimale dans un environnement de RL|La programmation dynamique est une méthode pour résoudre des problèmes de décision séquentielle en décomposant le problème en sous-problèmes plus simples et en utilisant une relation de récurrence entre ces sous-problèmes. Elle est souvent utilisée pour résoudre des MDPs discrets et déterministes|
|Méthodes de Monte Carlo|méthodes sont utilisées pour estimer la valeur d'une politique en échantillonnant des trajectoires complètes à partir de l'environnement. Elles sont particulièrement utiles lorsque les modèles de transition d'état ne sont pas connus||
|td learning|TD Learning est une famille d'algorithmes d'apprentissage par renforcement qui mettent à jour les estimations de valeur des états ou des actions en utilisant une combinaison de la méthode de Monte Carlo et de la méthode de la différence temporelle. Les méthodes TD sont souvent utilisées lorsque les transitions d'état sont partiellement observables||
|Réseaux de neurones profonds (Deep Q-Networks, DQN)|Les DQN sont une méthode qui utilise des réseaux de neurones profonds pour apprendre directement à partir de données brutes en entrée, tels que des pixels dans une image||
|Policy Gradients|Cette approche consiste à optimiser directement la politique d'action en utilisant des méthodes de descente de gradient. Elle est souvent utilisée lorsque les actions sont continues ou lorsque l'approximation de la fonction de valeur est difficile||
|...|||

## 2.  Implementing Snake with Q-learning
- Environment — Physical world in which the agent operates
- State — Current situation of the agent
- Reward — Feedback from the environment
- Policy — Method to map agent’s state to actions
- Value — Future reward that an agent would receive by taking an action in a particular state

[https://www.youtube.com/watch?v=L8ypSXwyBds](snake tuorial with neural networks)
[https://www.youtube.com/watch?v=ZhoIgo3qqLU](frozen lake gym tuto)






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

### **Sources**
- [Slides on RL by Pieter Abbeel and John Schulman - Open AI/ Berkeley AI Research Lab](https://people.eecs.berkeley.edu/~pabbeel/nips-tutorial-policy-optimization-Schulman-Abbeel.pdf)
- [Cours videos en 10 lessons by David Silver DeepMind](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- [Tutorial implementing Neural Network Policy Gradients for Pong game](https://karpathy.github.io/2016/05/31/rl/)
- [DeepMind Lab = open source 3D game-like platform created for agent-based AI research with rich simulated environments](https://deepmind.google/discover/blog/open-sourcing-deepmind-lab/)
- [Malmo, Microsoft = AI experimentation platform for supporting fundamental research in AI](https://www.microsoft.com/en-us/research/project/project-malmo/)
- [Gym OpenAI](https://www.gymlibrary.dev/index.html)

Intall VM
- ajouter Vbox Guest additions dans stockage
- config > general > avancé > presse-papier bidirectionnel
- sudo apt install linux-headers-$(uname -r) build-essential dkms
- redemarrer la vm
- [tuto](https://www.youtube.com/watch?v=MI1THQJFZXY)






