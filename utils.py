import matplotlib.pylab as plt

## Some fundamental constants
e           = 1.6021766208E-19 # C
e0          = 8.854187817E-12  # F m^{-1}
er          = 1.000589
air_density = 1.204            # kg/m^3 at 20 degree celsius and 1013.25 hPa
Wair        = 33.97            # average energy per ion pair in eV.

## Reference temperature and pressure
refTemperature = 293.15  # K
refPressure    = 1013.25 # hPa


def transport(charge, chargeNew, velocity, direction, area, rel):

    """
    Function to transport the charge densities in a given direction.
    """

    if direction == 1:
        chargeNew[1:] = charge[1:] + (charge[:-1] * velocity[:-1] * area[:-1] / area[1:] 
                                      - charge[1:] * velocity[1:]) * rel
        chargeNew[0]  -= chargeNew[0] * velocity[0] * rel
        return area[-1] * charge[-1] * velocity[-1]

    if direction == -1:
        chargeNew[:-1] = charge[:-1] + (charge[1:] * velocity[1:] * area[1:] / area[:-1] 
                                        - charge[:-1] * velocity[:-1]) * rel
        chargeNew[-1] -= chargeNew[-1] * velocity[-1] * rel
        return area[0] * charge[0] * velocity [0]

def plotFrame(ax, fig, x, nPosNew, nENew, nNegNew, eField, time):

    """
    Function for plotting.
    """

    ax[0].cla()
    ax[1].cla()

    ax[0].plot(x * 1E3, nPosNew, "-b", linewidth = 1.5, label = "Positive ions")
    ax[0].plot(x * 1E3,   nENew, "-k", linewidth = 1.5, label = "Electrons")
    ax[0].plot(x * 1E3, nNegNew, "-m", linewidth = 1.5, label = "Negative ions")

    ax[1].plot(x * 1E3, eField * 1E-3, "-k", linewidth = 1.5)

    fig.suptitle(fr"Time in simulation = {time * 1E6:.3f} $\upmu$s")

    ax[0].set_xlabel("Distance (mm)")
    ax[1].set_xlabel("Distance (mm)")

    ax[0].set_ylabel(r"Carrier densities (m$^{-3}$)")
    ax[1].set_ylabel(r"Electric field (kV\,m$^{-1}$)")
    
    fig.tight_layout(rect = [0, 0, 1, 1.05])

    leg = ax[0].legend(borderpad = 0.2, fontsize = 15)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_boxstyle('Square')

    plt.pause(1E-3)
