import os

import eagerpy as ep
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from secml.array import CArray
from secml.figure import CFigure
from secml.optim.constraints import CConstraintL1, CConstraintL2, CConstraintBox

from attacks.ddn_lp import L2DDNAttack, L1DDNAttack, LInfDDNAttack
from models.model_wrapper import ModelTracker
from utils.confidence_misclassification import TargetedMisclassificationThreshold

matplotlib.rc('text', usetex=True)

random_state = 999

n_features = 2  # Number of features
n_samples = 1250  # Number of samples
centers = [
    [-5, 5],
    [5, -5],
    [5, 5],
    [-5, -5],
    [0, 0],
    [2, 0]
]  # Centers of the clusters
n_classes = len(centers)
cluster_std = 0.8  # Standard deviation of the clusters

from secml.data.loader import CDLRandomBlobs

dataset = CDLRandomBlobs(n_features=n_features,
                         centers=centers,
                         cluster_std=cluster_std,
                         n_samples=n_samples,
                         random_state=random_state).load()

n_tr = 1000  # Number of training set samples
n_ts = 250  # Number of test set samples

# Split in training and test
from secml.data.splitter import CTrainTestSplit

splitter = CTrainTestSplit(
    train_size=n_tr, test_size=n_ts, random_state=random_state)
tr, ts = splitter.split(dataset)

# Normalize the data
from secml.ml.features import CNormalizerMinMax

nmz = CNormalizerMinMax()
tr.X = nmz.fit_transform(tr.X)
ts.X = nmz.transform(ts.X)

# Creation of the multiclass classifier
from torch import nn, optim
import torch

# Random seed
torch.manual_seed(0)

hidden = 20
model = nn.Sequential(nn.Linear(n_features, hidden),
                      nn.ReLU(),
                      nn.Linear(hidden, n_classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.001, momentum=0.9)

# wrap torch model in CClassifierPyTorch class
from secml.ml.classifiers import CClassifierPyTorch

clf = CClassifierPyTorch(model=model,
                         loss=criterion,
                         optimizer=optimizer,
                         input_shape=(n_features,),
                         random_state=0)

# Metric to use for training and performance evaluation
from secml.ml.peval.metrics import CMetricAccuracy

metric = CMetricAccuracy()

# We can now fit the classifier
clf.fit(tr.X, tr.Y)

# Compute predictions on a test set
y_pred = clf.predict(ts.X)

# Evaluate the accuracy of the classifier
acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)

print("Accuracy on test set: {:.2%}".format(acc))

pt_model = clf.model
# wrap model with foolbox
import foolbox as fb

bounds = (0, 1)
fmodel = fb.PyTorchModel(pt_model, bounds=bounds, preprocessing=None)
fmodel = fmodel.transform_bounds((0, 1))

n_pts = 1
images = torch.from_numpy(ts.X[:n_pts, :].tondarray()).float()
labels = torch.from_numpy(ts.Y[:n_pts].tondarray())
acc = fb.utils.accuracy(fmodel, images, labels)

fmodel = ModelTracker(fmodel, images)

p = np.inf
stepsize = 0.3
steps = 30
target = False
if target is not False:
    target_classes = (labels + 1) % n_classes * target
    criterion = fb.criteria.TargetedMisclassification(target_classes)
    # criterion = TargetedMisclassificationThreshold(target_classes, kappa=0.5)
else:
    criterion = fb.criteria.Misclassification(labels)
    target_classes = labels

use_init = False
if use_init:
    init_attack = fb.attacks.DatasetAttack()
    for pt in range(tr.X.shape[0]):
        init_attack.feed(fmodel, torch.from_numpy(tr.X[pt, :].tondarray()).float())
else:
    init_attack = None

fmodel.reset()
if p == 2:
    attack = L2DDNAttack(steps=steps, max_stepsize=stepsize, init_attack=init_attack, store_vars=True)
elif p == 1:
    attack = L1DDNAttack(steps=steps, max_stepsize=stepsize, init_attack=init_attack, store_vars=True)
elif p == np.inf:
    attack = LInfDDNAttack(steps=steps, max_stepsize=stepsize, init_attack=init_attack, store_vars=True)

import numpy as np

advs_ddn, _, is_adv_ddn = attack(fmodel, images, criterion, epsilons=np.inf)
path = np.vstack(attack._path)
epsilons = attack._epsilon
path = path.reshape(path.shape[0], path.shape[-1])
path = CArray(path)
stepsizes = attack._stepsizes

final_eps = abs(advs_ddn - images).sum()


def loss_fn(inputs, labels, model, targeted):
    inputs = torch.from_numpy(inputs.tondarray()).float()
    labels = ep.astensor(labels)

    logits = ep.astensor(model(inputs))

    if targeted is False:
        c_minimize = best_other_classes(logits, labels)
        c_maximize = labels  # target_classes
    else:
        c_minimize = labels  # labels
        c_maximize = best_other_classes(logits, labels)

    loss = logits[:, c_minimize] - logits[:, c_maximize]

    return -loss.sum().item()


def best_other_classes(logits, exclude):
    other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)


if isinstance(criterion, fb.criteria.TargetedMisclassification):
    is_adv_path = path[clf.predict(path) == target_classes.item(), :]
else:
    is_adv_path = path[clf.predict(path) != labels.item(), :]
eps_val = (advs_ddn - images).view(-1).norm(p=p).item()
if p == 2:
    eps_constraint = CConstraintL2(center=CArray(images.numpy()), radius=eps_val)
elif p == 1:
    eps_constraint = CConstraintL1(center=CArray(images.numpy()), radius=eps_val)
elif p == np.inf:
    eps_constraint = CConstraintBox(lb=CArray(images.numpy()) - eps_val,
                                    ub=CArray(images.numpy()) + eps_val)

image_path = "../../../images/"
fig = CFigure(width=5, height=5)

replot_bg = True
if replot_bg:
    n_grid_pts = 20

    # Convenience function for plotting the decision function of a classifier
    fig.sp.plot_decision_regions(clf, n_grid_points=200, plot_background=False)
    fig.sp.plot_fun(func=loss_fn,
                    multipoint=False,
                    colorbar=False,
                    cmap='coolwarm',
                    n_grid_points=n_grid_pts,
                    func_args=[target_classes, fmodel, target])
    fig.sp.grid(grid_on=False)
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1)
    fig.sp.xlim([0, 1])
    fig.sp.ylim([0, 1])
    plt.savefig(os.path.join(image_path, 'bg.png'),
                bbox_inches=0,
                format='png')
    plt.show()
    fig.close()

import matplotlib.image as mpimg

bg = mpimg.imread(os.path.join(image_path, 'bg.png'))
epsilons = epsilons.ravel()

for step in range(path.shape[0] - 1):
    if isinstance(attack, L2DDNAttack):
        eps_constraint = CConstraintL2(center=images, radius=epsilons[step])
    elif isinstance(attack, L1DDNAttack):
        eps_constraint = CConstraintL1(center=images, radius=epsilons[step])
    elif isinstance(attack, LInfDDNAttack):
        eps_constraint = CConstraintBox(lb=images-epsilons[step],
                                        ub=images+epsilons[step])
    else:
        raise NotImplementedError

    stepsize_constraint = CConstraintL2(center=path[step, :], radius=stepsizes[step])

    fig = CFigure(width=5, height=5)
    fig.sp.imshow(bg, extent=[0.0, 1.0, 0.0, 1.0])
    fig.sp.xlim([0, 1])
    fig.sp.ylim([0, 1])

    fig.sp.plot_constraint(eps_constraint)
    fig.sp.plot_constraint(stepsize_constraint)
    fig.sp.plot_path(path[:step + 2, :])
    fig.sp.scatter(path[step, 0], path[step, 1])
    fig.sp.grid(grid_on=False)
    fig.savefig(os.path.join(image_path, "{:03d}.png".format(step)))
    fig.close()

os.system(f"convert -delay 20 {image_path}/*.png {image_path}/path.gif")