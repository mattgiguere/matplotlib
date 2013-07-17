# -*- encoding: utf-8 -*-

from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, knownfailureif, cleanup
import os

@image_comparison(baseline_images=['pdf_use14corefonts'], extensions=['pdf'])
def test_use14corefonts():
    rcParams['pdf.use14corefonts'] = True
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 8
    rcParams['font.sans-serif'] = ['Helvetica']

    title = u'Test PDF backend with option use14corefonts=True'

    text = u'''A three-line text positioned just above a blue line
    and containing some French characters and the euro symbol:
    "Merci pépé pour les 10 €"'''

    plt.figure()
    plt.title(title)
    text = plt.text(0.5, 0.5, text, horizontalalignment='center',
             verticalalignment='bottom',
             fontsize=24)
    plt.axhline(0.5, linewidth=0.5)

@cleanup
def test_lookup():
    rcParams['pdf.use14corefonts'] = True
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 8
    rcParams['font.sans-serif'] = ['Helvetica']

    text = plt.text(0.5, 0.5, "Some text", horizontalalignment='center',
             verticalalignment='bottom',
             fontsize=24)
    from matplotlib import font_manager
    print(font_manager.findfont(text._fontproperties, fontext='afm'))
    filename = font_manager.findfont(
        text._fontproperties, fontext='afm', directory=os.path.join(
            rcParams['datapath'], 'fonts', 'pdfcorefonts'))
    print(filename)
    filename = font_manager.findfont(
            "Helvetica", fontext='afm',
            directory=os.path.join(
            rcParams['datapath'], 'fonts', 'pdfcorefonts'))
    print(filename)
    assert False


@cleanup
def test_type42():
    import io

    rcParams['pdf.fonttype'] = 42

    fig =  plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1,2,3])
    fig.savefig(io.BytesIO())
