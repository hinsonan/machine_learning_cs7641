import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve, ShuffleSplit

def plot_accuracy(pred, truth):
    #flatten the array inputs
    pred = pred.flatten()
    truth = truth.flatten()
    plt.plot(pred, label='predicted')
    plt.plot(truth, label='truth')

    # plt.set(xlabel='Index', ylabel='classification value',title='Accuracy')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

def plot_svm_iterative_learning_curve(scores, title):
    _, axes = plt.subplots(1)
    axes.plot(scores)
    axes.set_ylabel('Hinge Loss')
    axes.set_xlabel('epoch')
    axes.set_title("SVM Iterative Learning Curve")
    axes.grid()
    axes.legend(['training'], loc='upper left')
    plt.savefig(f'supervised_learning/charts/{title}')

def plot_confusion_matrix(pred, truth):
    conf_matrix = confusion_matrix(pred,truth)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

def plot_neural_net_history_accuracy(history, title):
    # summarize history for accuracy
    _, axes = plt.subplots(1)
    axes.plot(history['accuracy'])
    axes.plot(history['val_accuracy'])
    axes.set_title('model accuracy')
    axes.set_ylabel('accuracy')
    axes.set_xlabel('epoch')
    axes.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'supervised_learning/charts/{title}')

def plot_neural_net_history_loss(history, title):
    # summarize history for loss
    _, axes = plt.subplots(1)
    axes.plot(history['loss'])
    axes.plot(history['val_loss'])
    axes.set_title('model loss')
    axes.set_ylabel('loss')
    axes.set_xlabel('epoch')
    axes.legend(['train', 'test'], loc='upper right')
    plt.savefig(f'supervised_learning/charts/{title}')

def plot_multiple_histories(history1, history2, title, labels):
    _, axes = plt.subplots(2)
    plt.subplots_adjust(hspace=0.55)
    axes[0].plot(history1['accuracy'], label=f'{labels[0]} training')
    axes[0].plot(history1['val_accuracy'], label=f'{labels[0]} validation')
    axes[0].plot(history2['accuracy'], label=f'{labels[1]} training')
    axes[0].plot(history2['val_accuracy'], label=f'{labels[1]} validation')
    axes[0].set_title('model accuracy')
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('epoch')
    axes[0].legend(loc='upper left', prop={'size': 6})

    axes[1].plot(history1['loss'], label=f'{labels[0]} training')
    axes[1].plot(history1['val_loss'], label=f'{labels[0]} validation')
    axes[1].plot(history2['loss'], label=f'{labels[1]} training')
    axes[1].plot(history2['val_loss'], label=f'{labels[1]} validation')
    axes[1].set_title('model loss')
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    axes[1].legend(loc='upper left', prop={'size': 6})
    plt.savefig(f'supervised_learning/charts/{title}')

def plot_bar_graph(x_axis_labels: list, y_axis_labels: list):
    plt.bar(x_axis_labels,y_axis_labels)
    plt.title('Training Times in Seconds')
    plt.xlabel('Algorithm Type')
    plt.ylabel('Training Time')
    plt.show()

def plot_learning_curve(estimator, title, X, y, filename, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots()

    axes.set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, 
                       X, 
                       y, 
                       cv=cv,
                       scoring='accuracy', 
                       n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, 'o-')
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    # axes[2].grid()
    # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    plt.savefig(f'supervised_learning/charts/{filename}')


def plot_multiple_learning_curves(estimators, hyper_param_key, title, X, y, filename, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots()

    axes.set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Accuracy")
    dic = {}
    for estimator in estimators:
        dic[f'{estimator.__dict__[hyper_param_key]}'] = {'training_score':None, 'testing_sore':None}
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, 
                        X, 
                        y, 
                        cv=cv,
                        scoring='accuracy',
                        n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        dic[f'{estimator.__dict__[hyper_param_key]}']['training_score'] = train_scores_mean
        dic[f'{estimator.__dict__[hyper_param_key]}']['testing_score'] = test_scores_mean
    for key, val in dic.items():
    # Plot learning curve
        axes.plot(train_sizes, val['training_score'], 'o-',
                    label=f"Training score {hyper_param_key}: {key}")
    for key, val in dic.items():
        axes.plot(train_sizes, val['testing_score'], '^--',
                    label=f"Cross-validation score {hyper_param_key}: {key}")
    axes.legend(loc="upper left", prop={'size': 5})

    axes.grid()
    plt.savefig(f'supervised_learning/charts/{filename}')
    #plt.show()