from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from PIL import Image
import os

def show_plot_class_distribution_pie(class_distribution_df, class_column_name, total_column_name, title):
    Class_Id_Dist_Total = class_distribution_df.set_index(class_column_name)[total_column_name]
    
    fig = px.pie(
        names=Class_Id_Dist_Total.index, 
        values=Class_Id_Dist_Total.values,
        hole=0.5
    )

    annotations = dict(
            text='',
            font_size=12,
            showarrow=False,
        )
    fig.update_layout(
        title=title,
        font_size=15,
        title_x=0.45,
        annotations=[annotations]
    )
    fig.update_traces(textfont_size=15, textinfo='percent')
    fig.show()

def show_plot_class_distribution(class_distribution_df, x, y, title, figure_size):
    plt.figure(figsize=figure_size)
    sns.barplot(data=class_distribution_df, x=x, y=y, hue=x, palette="tab10")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.show()

def get_dims(file):
  im = Image.open(file)
  arr = np.array(im)

  if arr.ndim == 3:  # RGB or RGBA image
    h, w, _ = arr.shape  
  else: 
    h, w = arr.shape

  return h, w

def show_plot_image_sizes(data_path, label):
    label_path = os.path.join(data_path, label)
    images_list = os.listdir(label_path)

    # Process image dimensions without Dask
    dims = [get_dims(os.path.join(label_path, image)) for image in images_list]

    dim_df = pd.DataFrame(dims, columns=['height', 'width'])

    sizes = dim_df.groupby(['height', 'width']).size().reset_index().rename(columns={0: 'count'})

    sizes.plot.scatter(x='width', y='height')
    plt.title(f'Image Sizes (pixels) | {label}')
    plt.show()

def get_all_image_sizes(data_path, class_labels):
    all_sizes = []
    
    for label in class_labels:
        label_path = os.path.join(data_path, label)
        images_list = os.listdir(label_path)

        dims = [get_dims(os.path.join(label_path, image)) for image in images_list]
        dim_df = pd.DataFrame(dims, columns=['height', 'width'])
        dim_df['label'] = label

        all_sizes.append(dim_df)
    
    all_sizes_df = pd.concat(all_sizes, ignore_index=True)
    all_sizes_df = all_sizes_df.groupby(['height', 'width', 'label']).size().reset_index().rename(columns={0: 'count'})
    return all_sizes_df

def show_scatter_for_image_sizes(sizes_df, class_labels, xlim=None, ylim=None, figsize=(10, 6), title='Image Sizes (pixels) for All Classes'):
    plt.figure(figsize=figsize)
    
    for label in class_labels:
        label_sizes = sizes_df[sizes_df['label'] == label]
        plt.scatter(label_sizes['width'], label_sizes['height'], label=label, alpha=0.5)

    if xlim is not None and ylim is not None:
        plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]])

    plt.title(title)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.legend()
    plt.show()