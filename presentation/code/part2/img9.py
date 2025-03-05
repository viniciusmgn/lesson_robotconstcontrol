import numpy as np
import matplotlib.pyplot as plt

def plot_contour():
    """
    Plots contour curves of the function f(x,y) = x^4 + x^2.
    """
    # Define the function
    def f(x, y):
        return x*x+y*y*y*y   # Ensure variation in y
    
    # Create grid
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    # Set up the plot
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#191919')
    ax.set_facecolor('#191919')
    
    levels = [ (0.2*i)**2 for i in range(10) ]
    
    # Plot contour lines
    contour = ax.contour(X, Y, Z, levels=levels, colors=['cyan', 'magenta', 'yellow', 'lime', 'red', 'blue'])
    ax.clabel(contour, inline=True, fontsize=8, colors='white')
    
    # Customize axis
    ax.set_xlabel('x', color='white')
    ax.set_ylabel('y', color='white')
    ax.tick_params(colors='white')
    
    plt.show()

# Run the plot function
plot_contour()


#plt.savefig("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part1/image8.png", facecolor=bg_color)

    