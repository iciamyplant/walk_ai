import os
import pyvirtualdisplay
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
import pybullet_envs
import gym

# Démarrer le display virtuel
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()

# Arrêter et réinitialiser Ray
ray.shutdown()
ray.init(ignore_reinit_error=True)

# Définir l'environnement et l'enregistrement de l'environnement
ENV = 'HumanoidBulletEnv-v0'
def make_env(env_config):
    import pybullet_envs
    return gym.make('HumanoidBulletEnv-v0')

# Définir le nombre cible de récompense (tu peux changer à 2000 je pense que ça fonctionne quand même)
TARGET_REWARD = 6000

# Définir le modèle d'agent
TRAINER = PPOTrainer

# Chemin du dossier du script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Créer un dossier de résultats dans le dossier du script
results_dir = os.path.join(script_dir, "results/3D")
os.makedirs(results_dir, exist_ok=True)

# Définir un callback personnalisé pour enregistrer la vidéo tous les 100 epochs
class VideoLogger(DefaultCallbacks):
    def on_train_result(self, *, trainer, result, **kwargs):
        epoch = result["training_iteration"]
        if epoch % 100 == 0:
            video_path = os.path.join(results_dir, f"video_epoch_{epoch}.mp4")
            trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(lambda env: env.unwrapped.save_video(video_path))
            )

# Lancer l'entraînement avec le paramètre local_dir spécifié et le callback personnalisé
tune.run(
    TRAINER,
    stop={"episode_reward_mean": TARGET_REWARD},
    config={
        "env": ENV,
        "num_workers": 3, #modifier ici en fonction du nombre de cpu (nb de cpus - 1)
        "num_gpus": 0, #modifier ici en fonction du nombre de gpus
        "monitor": True,
        "evaluation_num_episodes": 50,
        "gamma": 0.995,
        "lambda": 0.95,
        "clip_param": 0.2,
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "lr": .0001,
        "sgd_minibatch_size": 32768,
        "train_batch_size": 320_000,
        "model": {
            "free_log_std": True,
        },
        "batch_mode": "complete_episodes",
        "observation_filter": "MeanStdFilter",
        "callbacks": VideoLogger  # Utiliser le callback personnalisé
    },
    local_dir=results_dir  # Spécifier l'emplacement où seront enregistrés les résultats
)
