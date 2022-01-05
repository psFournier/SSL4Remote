# Guide complet pour faire tourner des expes from scratch

## Etape 1 : accès gitlab, clone du repo, création de branche

### Configurer son accès gitlab depuis un réseau extérieur au CNES  (ordi perso)

https://confluence.cnes.fr/pages/viewpage.action?pageId=26159013

### Configurer son accès gitlab depuis le réseau interne du CNES (pour le HPC)

https://confluence.cnes.fr/pages/viewpage.action?pageId=26166114

### Créer une nouvelle branche de travail personnel

Sur un de ses deux postes de travail entre perso et HPC, faire:

`git checkout -b branche_perso` : crée une nouvelle branche locale et se 
positionne dessus ;

puis `git push -u origin branche_perso` : push la branche locale sur le remote 
repository en précisant son "upstream" vu qu'elle n'y existe pas encore. 
origin est le nom du répertoire disant (peut-être autre chose que origin) ;

enfin on synchronise cette opération sur l'autre poste de travail depuis 
cet autre poste de travail avec `git 
fetch origin` puis `git checkout branche_perso`.

A cette étape, les deux machines sont sur leur branche locale et suivent la 
même branche distante.

## Etape 2 : création de l'environnement virtuel, installation des packages

Sur le HPC, dans le répertoire du repo cloné, faire:
`module load python/3.7.2`
puis
`virtualenv venv`

Sur une machine perso, dans le répertoire `virtualenv venv` suffit si le 
package virtualenv est installé. Sinon faire `python3 -m pip install --user virtualenv`

Sur les deux machines, une fois l'environnement virtuel créé, faire 
`venv/bin/pip install --upgrade pip` puis 
`venv/bin/pip install -r requirements.txt` pour installer les packages 
nécessaires. Le fichier requirements.txt est supposé à jour...

## Etape 3 : scripts de lancement d'expés

Une fois que tout est en place, on peut lancer un entraînement de test avec scripts/train_local.sh après l'avoir rendu éxécutable (chmod +x ...)
Les paramètres: --batch_size 4 --workers 0 --max_epochs 5 --limit_train_batches 5 
--limit_val_batches 2 sont là pour permettre un test rapide du code, il faut évidemment les enlever pour des entraînements complets.


### Lancer des expés sur le cluster HPC

Il est nécessaire sur le cluster de lancer ses expés en passant au 
gestionnaire de jobs PBS un script appelant `train.py`, afin que les ressources 
disponibles soit 
optimisées. Pour cela, il suffit d'utiliser le script 
`dl_toolbox/scripts/train_hal_venv.sh` en faisant `qsub dl_toolbox/scripts/train_hal_venv.sh`.

Qsub retourne alors le numéro du job lancé N. On peut suivre son avancement 
avec `qpeek N`, et voir la liste des jobs lancés et leur statut avec `qstat 
-u <nom_utilisateur>`. Plus d'options sont détaillées [ici](https://gitlab.cnes.fr/hpc/wikiHPC/-/wikis/home#usage-orient%C3%A9-calcul-et-d%C3%A9veloppement).

## Etape 4 : visualisation et postprocessing, JupyterHub Desktop

Une fois que les scripts ont tourné, les logs se trouvent dans 
`outputs/tensorbaord`. Pour pouvoir les lire avec l'interface graphique 
Tensorboard (impossible sur HAL directement), le mieux est de passer par le 
Virtual Research 
Environment (VRE)
du projet. Il faut se connecter sur jupyterhub.cnes.fr avec les mêmes 
identifiants que sur HAL puis choisir le noeud de visu par défaut. Une fois 
le serveur lancé, il faut ouvrir un desktop pour se retrouver avec une 
interface de type Windows mais l'accès au fichier sur le cluster. A partir 
de là, ouvrir un terminal, se placer dans son répertoire semisup 
(/home/eh/fournip/SemiSupervised/SSL4Remote pour moi) et lancer `tensorboard 
--logdir=outputs/tensorboard`. Ensuite se connecter à l'adresse indiquée par 
tensorboard avec firefox et _voila_.
