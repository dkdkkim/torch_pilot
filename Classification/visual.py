import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter

def plot_training(training_losses, validation_losses, learning_rate, gausian=True, sigma=2, figsize=(8,6)):
    """
    Returns a loss plot with traingin loss, validation loss and learning rate.
    """

    list_len = len(training_losses)
    x_range = list(range(1, list_len+1))

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    subfig1 = fig.add_subplot(grid[0,0])
    subfig2 = fig.add_subplot(grid[0,1])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, 1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    if gausian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gause = gaussian_filter(validation_losses, sigma=sigma)

        linestyle_original = '.'
        color_original_train = 'lightcoral'
        color_original_valid = 'lightgreen'
        color_smooth_train = 'red'
        color_smooth_valid = 'green'
        alpha = 0.25
    else:
        linestyle_original = '-'
        color_original_train = 'red'
        color_original_valid = 'green'
        alpha = 1.0

    subfig1.plot(x_range, training_losses, linestyle_original, color=color_original_train, label='Training',
                 alpha=alpha)
    subfig1.plot(x_range, validation_losses, linestyle_original, color=color_original_valid, label='Validation',
                 alpha=alpha)
    if gausian:
        subfig1.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig1.plot(x_range, validation_losses_gause, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig1.title.set_text('Training & validation loss')
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('Loss')

    subfig1.legend(loc='upper right')

    subfig2.plot(x_range, learning_rate, color='black')
    subfig2.title.set_text('Learning rate')
    subfig2.set_xlabel('Epoch')
    subfig2.set_ylabel('LR')
    return fig

def fig_3views_with_lbl(img, lbl, is_uint=False, **kwargs):
    import matplotlib
    matplotlib.use(kwargs['mlpBack'])
    import matplotlib.pyplot as plt

    if is_uint:
        clim_max = 255.
    else:
        clim_max = 1.
    rows = 4

    fig, ax = plt.subplots(rows, 6, figsize=(12, rows*2))
    for pidx in range(4):
        ax[pidx, 0].imshow(img[img.shape[0] / 2 - rows/2 + pidx,:,:],
                                      cmap='bone', clim=(0., clim_max))  # , clim=(0., 1.5))
        ax[pidx, 1].imshow(lbl[lbl.shape[0] / 2 - rows/2 + pidx,:,:],
                                      cmap='bone', clim=(0., 1.0))  # , clim=(0., 1.5))

        ax[pidx, 2].imshow(img[:, img.shape[1] / 2 - rows/2 + pidx, :],
                                          cmap='bone', clim=(0., clim_max))  # , clim=(0., 1.5))
        ax[pidx, 3].imshow(lbl[:, lbl.shape[1] / 2 - rows/2 + pidx, :],
                                          cmap='bone', clim=(0., 1.0))  # , clim=(0., 1.5))

        ax[pidx, 4].imshow(img[:, :, img.shape[2] / 2 - rows/2 + pidx],
                                          cmap='bone', clim=(0., clim_max))  # , clim=(0., 1.5))
        ax[pidx, 5].imshow(lbl[:, :, lbl.shape[2] / 2 - rows/2 + pidx],
                                          cmap='bone', clim=(0., 1.0))  # , clim=(0., 1.5))
        ax[pidx % 4, 4 + pidx / 4].axis('off')

    for a in ax:
        for aa in a:
            aa.axis('off')

    ax[0, 0].text(1, 4, '<transverse>', color='white')
    ax[0, 2].text(1, 4, '<coronal>', color='white')
    ax[0, 4].text(1, 4, '<sagittal>', color='white')
