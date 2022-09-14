import numpy as np
import matplotlib.pyplot as plt

def band_ploter(data, title = 'TR por bandas'):
    bars = ('125', '250', '500', '1000', '2000', '4000', '8000')
    y_pos = np.arange(len(bars))

    # Create bars
    plt.bar(y_pos, data)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Tr [s]')
    plt.title(title)
    # Create names on the x-axis
    plt.xticks(y_pos, bars)

    # Show graphic
    plt.show()