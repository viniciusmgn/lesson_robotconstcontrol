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
    
    lvl_min = levels[0]
    lvl_max = levels[-1]
    
    # Plot contour lines
    color_level=[]
    for lvl in levels:
        x = (lvl-lvl_min)/(lvl_max-lvl_min)
        x=1-x
        value_red = int(155*x)
        value_green = int(100+155*x)
        value_blue = int(155*x)
        color_level.append( f"#{value_red:02X}{value_green:02X}{value_blue:02X}" )
    
    contour = ax.contour(X, Y, Z, levels=levels, colors=color_level)
    ax.clabel(contour, inline=True, fontsize=8, colors='white')
    
    # Customize axis
    ax.set_xlabel('$x_1$', color='white', fontsize=16)
    ax.set_ylabel('$x_2$', color='white', fontsize=16)
    ax.tick_params(colors='white')
    
    plt.show()

# Run the plot function
plot_contour()


#plt.savefig("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part1/image8.png", facecolor=bg_color)

    