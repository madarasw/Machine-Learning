import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.cluster import MiniBatchKMeans
from skimage import io
from ipywidgets import interact, IntSlider

img_dir = 'images/'


@interact
def color_compression(image=os.listdir(img_dir), k=IntSlider(min=1,
                                                             max=256,
                                                             step=1,
                                                             value=16,
                                                             continuous_update=False,
                                                             layout=dict(width='100%'))):
    image_path = img_dir+image
    input_img = io.imread(image_path)
    img_data = (input_img / 255.0).reshape(-1, 3)

    kmeans = MiniBatchKMeans(k).fit(img_data)
    k_colors = kmeans.cluster_centers_[kmeans.predict(img_data)]

    # load the large image into program
    # replace each of its pixels with the nearest of the centroid colors found from the small image.
    k_img = np.reshape(k_colors, (input_img.shape))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('K-means Image Compression', fontsize=20)

    ax1.set_title('Compressed')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(k_img)

    ax2.set_title('Original (16,777,216 colors)')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(input_img)

    plt.subplots_adjust(top=0.85)
    plt.show()


color_compression()
