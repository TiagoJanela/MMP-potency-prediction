from typing import *

import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw

DEFAULT_FONT = r".\..\ML\fonts\Arial.ttf"
DEFAULT_FONT_BOLD = r".\..\ML\fonts\Arial_Bold.ttf"
DEFAULT_FONT_ITALIC = r".\..\ML\fonts\Arial_Italic.ttf"


def replace_na_with_empty_image(images: np.ndarray):
    """
    Replace np.nan values in images with an empty PIL.Image object.
    :param images: matrix of images
    :return:
    """
    if np.any(pd.isnull(images)):
        i, j = np.where(pd.isnull(images))
        for ii, jj in zip(i, j):
            images[ii, jj] = Image.new(mode='RGB', size=(0, 0), color='white')
    return images


# General functions for (PIL) images
def get_grid_image(images: Union[np.ndarray, list]) -> Image:
    """
    Generate grid image with all given images.
    :param images: matrix with ncols and nrows of images. Grid will have the exact same dimensions
    :return: grid image
    """
    if isinstance(images, list):
        images_list = images
        images = np.empty((len(images), len(images[0])), dtype=object)
        for i in range(len(images_list)):
            for j in range(len(images_list[0])):
                images[i][j] = images_list[i][j]
    assert isinstance(images, np.ndarray), f"The given images have invalid data type {type(images)}."
    images = np.array(images, dtype=object)

    images = replace_na_with_empty_image(images)

    df_images = pd.DataFrame(images)

    df_images_sizes = df_images.applymap(lambda x: x.size)

    series_widths = df_images_sizes.apply(lambda x: max(map(lambda x: x[0], x)))
    series_heights = df_images_sizes.apply(lambda x: max(map(lambda x: x[1], x)), axis=1)

    total_width = series_widths.sum()
    total_height = series_heights.sum()

    grid = Image.new(mode='RGB', size=(total_width, total_height), color='white')
    grid.__setattr__('text', {})

    # Determine positions from given parameters
    positions = [(width, height)
                 for i, width in enumerate([0] + list(series_widths.cumsum().values)[:-1])
                 for j, height in enumerate([0] + list(series_heights.cumsum().values)[:-1])]
    for i, img in enumerate(images.ravel(order='F')):
        grid.paste(img, positions[i])

    return grid


def get_text_image(title: str, font: ImageFont.FreeTypeFont, size: Union[None, Tuple[int, int]] = None,
                   pos: Tuple[int, int] = (0, 0), width_spacing: int = 0, height_spacing: int = 0) -> Image.Image:
    """
    Generate an image with the given text
    :param title: text to display in the image
    :param size: size of the image, if not given: will generate minimal necessary size of the image to place the text
    :param pos: position of the text (upper left)
    :param font: font to write the text with
    :param width_spacing: additional width spacing
    :param height_spacing: additional height spacing
    :return:
    """
    if size is None:
        size = get_text_size(title, font=font)
    size = (size[0] + width_spacing, size[1] + height_spacing)
    img = Image.new("RGB", size, color='white')
    place_text_in_image(img, title, pos, font)
    img.__setattr__('text', {})
    return img


def place_text_in_image(img, text, pos, font=None, **kwargs):
    """
    Place text in an image
    :param img: image to place the text in
    :param text: text to place
    :param pos:
    :param fontsize:
    :param font:
    :param kwargs:
    :return:
    """
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, (0, 0, 0), font=font, **kwargs)


def get_text_size(text: str, font) -> Tuple[int, int]:
    """
    Return minimal necessary size of the image to place the text
    :param text:
    :param font:
    :return:
    """
    left, top, right, bottom = font.getbbox(text)
    width_text = right - left
    height_text = bottom

    n_lines = len(text.split('\n'))
    height_text = height_text * n_lines

    return width_text, height_text


def get_font(fontsize: int, fontname: str = DEFAULT_FONT) -> ImageFont.FreeTypeFont:
    """
    Returns the ImageFont.FreeFont in the given font size with the given font name
    :param fontsize: font size in pt
    :param fontname: path to the ttf-file of the font
    :return:
    """
    return ImageFont.truetype(fontname, fontsize)
