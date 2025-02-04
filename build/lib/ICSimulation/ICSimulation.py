import numpy            as np
import matplotlib.pylab as plt
import os

from .             import utils as u
from .ion_mobility import mob_pos, mob_neg

def CICpulsedSimulation(dpp, pulseDuration, alpha, voltage, temperature, pressure,
                        humidity, r1, r2, h, n, Ndw, figures = 0, eFieldP = 1,
                         timeStruct = [], doseRateStruct = []):

    """
    This function simulates the charge transport in a 1D cylindrical ionization
    chamber.

    *-- Inputs:
        
        :dpp:           Dose per pulse in Gy.
        :pulseDuration: Pulse duration in s.
        :alpha:         Volume recombination constant in m^{3} s^{-1}.
        :voltage:       Voltage in V.
        :temperature:   Temperature in degree Celsius.
        :pressure:      Pressure in hPa.
        :humidity:      Relative humidity in %.
        :r1:            Internal radii of the ionization chamber in m.
        :r2:            External radii of the ionization chamber in m.
        :h:             Heigh of the ionization chamber in m.
        :n:             Number of elements in which the geometry is divided.
        :Ndw:           Calibration coefficient in Gy C^{-1}. The calibration
                        coefficient must include all the correction factors
                        that affect the released charge excluding k_TP.
        :figures:       Optional. Set to 1 to have a graphical display of the
                        charge carrier densities while simulating.
        :eFieldP:       Optional. Set to 0 to disable electric field
                        perturbation.
        :timeStruct:    Optional, array with the time in s for a custom pulse
                        structure.
        :DoseStruct:    Optional, array with the dose rate in arbitrary units
                        for a custom pulse structure.

    *-- Returns:

        :CCE:    Charge collection efficiency.
        :FEF0:   Free electron fraction in relation to the released charge in the medium.
        :FEF1:   Free electron fraction in relation to the collected charge.
        :Q_col:  Collected charge per pulse referenced to 20 ºC and 1013.25 hPa.
        :I:      Array with the time in s and intensity in A for each of the
                 three species considered.
                    . I[:, 0] -> Array with the times in s.
                    . I[:, 1] -> Array with the instantaneous current from
                                 electrons in A.
                    . I[:, 2] -> Array with the instantaneous current from
                                 positive ions in A.
                    . I[:, 3] -> Array with the instantaneous current from
                                 negative ions in A.
    """

    # Chamber dimension properties.
    dx     = (r2 - r1) / n                                      # m
    x      = np.array([r1 + dx / 2 + i * dx for i in range(n)]) # m
    area   = 2 * np.pi * h * x                                  # m^2
    volume = np.pi * (r2**2 - r1**2) * h                        # m^3
    length = r2 - r1                                            # m    

    # Unperturbed electric field
    eField0 = abs(voltage) / (x * np.log(r2 / r1)) # V/m

    # Sign of the voltage
    eSign = np.sign(voltage)
 
    pars = [dpp, pulseDuration, alpha, voltage, temperature, pressure, humidity,
            area, volume, length, eField0, dx, x, eSign, n, Ndw, figures, eFieldP,
            timeStruct, doseRateStruct]
    
    chargePos, chargeNeg, chargeE, I = pulsedSimulation(*pars)

    k_TP = (273.15 + temperature) / u.refTemperature * u.refPressure / pressure

    # Charge released in the medium
    n0 = dpp / (u.e * Ndw * volume * k_TP) # m^-3

    CCE   = chargePos / (n0 * volume)  # Charge collection efficiency
    FEF0  = chargeE   / (n0 * volume)  # FEF (Boag definition 0)
    FEF1  = chargeE   / chargePos      # FEF (Boag definition 1)
    Qcoll = chargePos * u.e * k_TP     # Charge referenced to STP

    return CCE, FEF0, FEF1, Qcoll, I

def SICpulsedSimulation(dpp, pulseDuration, alpha, voltage, temperature, pressure,
                        humidity, r1, r2, n, Ndw, figures = 0, eFieldP = 1,
                        timeStruct = [], doseRateStruct = []):
    """
    This function simulates the charge transport in a 1D spherical ionization
    chamber.

    *-- Inputs:
        
        :dpp:           Dose per pulse in Gy.
        :pulseDuration: Pulse duration in s.
        :alpha:         Volume recombination constant in m^{3} s^{-1}.
        :voltage:       Voltage in V.
        :temperature:   Temperature in degree Celsius.
        :pressure:      Pressure in hPa.
        :humidity:      Relative humidity in %.
        :r1:            Internal radii of the ionization chamber in m.
        :r2:            External radii of the ionization chamber in m.
        :n:             Number of elements in which the geometry is divided.
        :Ndw:           Calibration coefficient in Gy C^{-1}. The calibration
                        coefficient must include all the correction factors
                        that affect the released charge excluding k_TP.
        :figures:       Optional. Set to 1 to have a graphical display of the
                        charge carrier densities while simulating.
        :eFieldP:       Optional. Set to 0 to disable electric field
                        perturbation.
        :timeStruct:    Optional, array with the time in s for a custom pulse
                        structure.
        :DoseStruct:    Optional, array with the dose rate in arbitrary units
                        for a custom pulse structure.

    *-- Returns:

        :CCE:    Charge collection efficiency.
        :FEF0:   Free electron fraction in relation to the released charge in the medium.
        :FEF1:   Free electron fraction in relation to the collected charge.
        :Q_col:  Collected charge per pulse referenced to 20 ºC and 1013.25 hPa.
        :I:      Array with the time in s and intensity in A for each of the
                 three species considered.
                    . I[:, 0] -> Array with the times in s.
                    . I[:, 1] -> Array with the instantaneous current from
                                 electrons in A.
                    . I[:, 2] -> Array with the instantaneous current from
                                 positive ions in A.
                    . I[:, 3] -> Array with the instantaneous current from
                                 negative ions in A.
    """

    # Chamber dimension properties.
    dx     = (r2 - r1) / n                                      # m
    x      = np.array([r1 + dx / 2 + i * dx for i in range(n)]) # m
    area   = 4 * np.pi * x**2                                   # m^2
    volume = 4 / 3 * np.pi * (r2**3 - r1**3)                    # m^3
    length = r2 - r1                                            # m    

    # Unperturbed electric field
    eField0 = - abs(voltage) / (x**2 * (1 / r2 - 1 / r1)) # V/m

    # Sign of the voltage
    eSign = np.sign(voltage)
 
    pars = [dpp, pulseDuration, alpha, voltage, temperature, pressure, humidity,
            area, volume, length, eField0, dx, x, eSign, n, Ndw, figures, eFieldP,
            timeStruct, doseRateStruct]
    
    chargePos, chargeNeg, chargeE, I = pulsedSimulation(*pars)

    k_TP = (273.15 + temperature) / u.refTemperature * u.refPressure / pressure

    # Charge released in the medium
    n0 = dpp / (u.e * Ndw * volume * k_TP) # m^-3

    CCE   = chargePos / (n0 * volume)  # Charge collection efficiency
    FEF0  = chargeE   / (n0 * volume)  # FEF (Boag definition 0)
    FEF1  = chargeE   / chargePos      # FEF (Boag definition 1)
    Qcoll = chargePos * u.e * k_TP     # Charge referenced to STP

    return CCE, FEF0, FEF1, Qcoll, I

def PPICpulsedSimulation(dpp, pulseDuration, alpha, voltage, temperature, pressure,
                         humidity, d, radii, n, Ndw, figures = 0, eFieldP = 1, 
                         timeStruct = [], doseRateStruct = []):
    
    """
    This function simulates the charge transport in a 1D parallel plate ionization
    chamber.

    *-- Inputs:
        
        :dpp:           Dose per pulse in Gy.
        :pulseDuration: Pulse duration in s.
        :alpha:         Volume recombination constant in m^{3} s^{-1}.
        :voltage:       Voltage in V.
        :temperature:   Temperature in degree Celsius.
        :pressure:      Pressure in hPa.
        :humidity:      Relative humidity in %.
        :d:             Distance between electrodes in m.
        :radii:         Radii of the sensitive volume of the ionization chamber
                        in m.
        :n:             Number of elements in which the geometry is divided.
        :Ndw:           Calibration coefficient in Gy C^{-1}. The calibration
                        coefficient must include all the correction factors
                        that affect the released charge excluding k_TP.
        :figures:       Optional. Set to 1 to have a graphical display of the
                        charge carrier densities while simulating.
        :eFieldP:       Optional. Set to 0 to disable electric field
                        perturbation.
        :timeStruct:    Optional, array with the time in s for a custom pulse
                        structure.
        :DoseStruct:    Optional, array with the dose rate in arbitrary units
                        for a custom pulse structure.

    *-- Returns:

        :CCE:    Charge collection efficiency.
        :FEF0:   Free electron fraction in relation to the released charge in the medium.
        :FEF1:   Free electron fraction in relation to the collected charge.
        :Q_col:  Collected charge per pulse referenced to 20 ºC and 1013.25 hPa.
        :I:      Array with the time in s and intensity in A for each of the
                 three species considered.
                    . I[:, 0] -> Array with the times in s.
                    . I[:, 1] -> Array with the instantaneous current from
                                 electrons in A.
                    . I[:, 2] -> Array with the instantaneous current from
                                 positive ions in A.
                    . I[:, 3] -> Array with the instantaneous current from
                                 negative ions in A.
    """

    # Chamber dimension properties.
    dx     = d / n                                         # m
    x      = np.array([dx / 2 + i * dx for i in range(n)]) # m
    area   = np.pi * radii**2 * np.ones(n)                 # m^2
    volume = np.pi * radii**2 * d                          # m^3
    length = d                                             # m    

    # Unperturbed electric field
    eField0 = abs(voltage) * np.ones(n) / d # V/m

    # Sign of the voltage
    eSign = 1
 
    pars = [dpp, pulseDuration, alpha, voltage, temperature, pressure, humidity,
            area, volume, length, eField0, dx, x, eSign, n, Ndw, figures, eFieldP, 
            timeStruct, doseRateStruct]
    
    chargePos, chargeNeg, chargeE, I = pulsedSimulation(*pars)

    k_TP = (273.15 + temperature) / u.refTemperature * u.refPressure / pressure
    
    # Charge released in the medium
    n0 = dpp / (u.e * Ndw * volume * k_TP) # m^-3

    CCE   = chargePos / (n0 * volume)  # Charge collection efficiency
    FEF0  = chargeE   / (n0 * volume)  # FEF (Boag definition 0)
    FEF1  = chargeE   / chargePos      # FEF (Boag definition 1)
    Qcoll = chargePos * u.e * k_TP     # Charge referenced to STP

    return CCE, FEF0, FEF1, Qcoll, I

def pulsedSimulation(dpp, pulseDuration, alpha, voltage, temperature, pressure,
                     humidity, area, volume, length, eField0, dx, x, eSign, n,
                     Ndw, figures = 0, eFieldP = 1, timeStruct = [],
                     doseRateStruct = []):
    
    """
    This function generically simulates the charge transport in a 1D ionization
    chamber. The parameters area and eField0 will essentially determine the
    symmetry of the ionziation chamber.

    *-- Inputs:
        
        :dpp:           Dose per pulse in Gy.
        :pulseDuration: Pulse duration in s.
        :alpha:         Volume recombination constant in m^{3} s^{-1}.
        :voltage:       Voltage in V.
        :temperature:   Temperature in degree Celsius.
        :pressure:      Pressure in hPa.
        :humidity:      Relative humidity in %.
        :area:          Area of the elements m^2. This parameter must be an
                        array.
        :volume:        Ionization chamber volume in m^-3.
        :length:        Length of the geometry in m.
        :eField0:       Initial electric field in V/m. This parameters must be 
                        an array.
        :dx:            Distance between the elements.
        :x:             Coordinates.
        :eSign:         Direction of the charge transport.
        :n:             Number of elements in which the geometry is divided.
        :Ndw:           Calibration coefficient in Gy C^{-1}. The calibration
                        coefficient must include all the correction factors
                        that affect the released charge excluding k_TP.
        :figures:       Optional. Set to 1 to have a graphical display of the
                        charge carrier densities while simulating.
        :eFieldP:       Optional. Set to 0 to disable electric field
                        perturbation.
        :timeStruct:    Optional, array with the time in s for a custom pulse
                        structure.
        :DoseStruct:    Optional, array with the dose rate in arbitrary units
                        for a custom pulse structure.

    *-- Returns:

        :chargePos:  "Collected" charge from positive ions in C.
        :chargeNeg:  "Collected" charge from negative ions in C.
        :chargeE:    "Collected" charge from electron in C.
        :I:          Array with the time in s and intensity in A for each of the
                     three species considered.
                         . I[:, 0] -> Array with the times in s.
                         . I[:, 1] -> Array with the instantaneous current from
                                      electrons in A.
                         . I[:, 2] -> Array with the instantaneous current from
                                      positive ions in A.
                         . I[:, 3] -> Array with the instantaneous current from
                                      negative ions in A.
    """

    # Negative and positive ion mobilities
    k_neg = mob_neg(humidity, pressure, temperature)
    k_pos = mob_pos(humidity, pressure, temperature)

    k_TP = (273.15 + temperature) / u.refTemperature * u.refPressure / pressure
    
    # Load electron properties
    c_dir  = os.path.dirname(__file__)
    import sys
    #eTable = np.loadtxt(f"{c_dir}/../data/dataElectrons.txt")
    file_path = os.path.join(sys.prefix, 'data')
    eTable    = np.loadtxt(f"{file_path}/dataElectrons.txt")

    # Temperature and pressure scaling rules.
    eTable[:, 0] /= k_TP 
    eTable[:, 2] /= k_TP 

    # Charge density arrays.
    nE   = np.zeros(n); nENew   = np.zeros(n)
    nPos = np.zeros(n); nPosNew = np.zeros(n)
    nNeg = np.zeros(n); nNegNew = np.zeros(n)

    # Charge released in the medium
    timeStructFlag = 0
    if len(timeStruct) > 0: 
        timeStructFlag  = 1
        doseRateStruct *= dpp / np.trapz(doseRateStruct, timeStruct)
        n0Struct        = doseRateStruct / (u.e * Ndw * volume * k_TP) # m^-3
        pulseDuration   = timeStruct[-1]
        n0              = np.trapz(n0Struct, timeStruct)

    else:
        n0 = dpp / (u.e * Ndw * volume * k_TP) # m^-3

    # Perturbed electric field
    eField  = eField0 # V/m

    # In case of pulse duration equal to zero, instantanous release of the charge
    # is assumed.
    if pulseDuration == 0.:
        nENew   += n0
        nPosNew += n0

    # Initialize some needed parameters
    tStep = 1E-13              # s
    time  = 0.0                # s
    nSum  = (2 * n0 * volume)  # dummy number to enter the while bucle
    index = 0     

    # "Collected" charges
    chargeE = 0.0; chargePos = 0.0; chargeNeg = 0.0
    # Instantaneous induced current
    Ie = []; Ipos = []; Ineg = []; t = []

    # If figures are enabled, create a canvas
    if figures == 1:
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

    # Loop until no charge (nSum > 1) is left in the simulation.
    while (nSum / (2 * n0 * volume) > 1E-10) or (time < pulseDuration):
        
        # Update time variables
        time  += tStep
        rel    = tStep / dx
        index += 1

        # Update transport coefficients:
        eVelocity   = np.interp(eField, eTable[:, 0], eTable[:, 1])
        eIRate      = np.interp(eField, eTable[:, 0], eTable[:, 2])
        # Electron attachment parametrization from Boissonnat et al.
        # (arXiv:1609.03740v1)
        eAttachment = 95.24E-9 * (1 - np.exp(-eField / 258.5E3))

        # Avoid non-physical negative charge distributions.
        nPosNew[nPosNew < 1E-30] = 0.0
        nENew[nENew < 1E-30]     = 0.0
        nNegNew[nNegNew < 1E-30] = 0.0

        # Update charge vectors
        nPos = nPosNew
        nNeg = nNegNew
        nE   = nENew

        # Charge release in the medium.
        if time <= pulseDuration and pulseDuration > 0:
            if timeStructFlag:
                nENew   += np.interp(time, timeStruct, n0Struct) * tStep
                nPosNew += np.interp(time, timeStruct, n0Struct) * tStep
            else:
                nENew   += n0 * tStep / pulseDuration
                nPosNew += n0 * tStep / pulseDuration

        # Recombination
        nNegNew -= alpha * nNeg * nPos * tStep
        nPosNew -= alpha * nNeg * nPos * tStep

        # Electric field perturbation
        if eFieldP:
            eField  = u.e / (u.er * u.e0 * area) * eSign * np.cumsum((nPos - nE - nNeg) * area * dx)
            eField += (abs(voltage) - np.sum(eField * dx)) * eField0 / abs(voltage)
            if np.any(eField < 0.0): raise Exception("Electric field is < 0.0")

        # Transport of the species
        chargePos += u.transport(nPos, nPosNew, k_pos * eField,  eSign, area, rel) * tStep
        chargeNeg += u.transport(nNeg, nNegNew, k_neg * eField, -eSign, area, rel) * tStep
        chargeE   += u.transport(  nE,   nENew,      eVelocity, -eSign, area, rel) * tStep

        # Attachment and electron multiplication
        nNegNew += nE * tStep / eAttachment
        nENew   += nE * (-1 / eAttachment + eIRate * eVelocity) * tStep
        nPosNew += nE * eIRate * eVelocity * tStep

        eSum = np.sum(nENew * area * dx)

        # Update time step
        if eSum / (2 * n0 * volume) > 1E-10: tStep = 0.4 * dx / max(np.max(eVelocity), np.max(k_neg * eField))
        else: tStep = 0.4 * dx / np.max(k_neg * eField)

        nSum = np.sum((nPosNew + nENew + nNegNew) * area * dx)

        # Compute instantanous induced current
        Ie.append(u.e * np.sum(eVelocity * nE * area * dx) / length)
        Ipos.append(u.e * np.sum(eField * k_pos * nPos * area * dx) / length) 
        Ineg.append(u.e * np.sum(eField * k_neg * nNeg * area * dx) / length)
        t.append(time)

        if (figures == 1) and (index%10 == 0):
            u.plotFrame(ax, fig, x, nPosNew, nENew, nNegNew, eField, time)

    return chargePos, chargeNeg, chargeE, np.array([t, Ie, Ipos, Ineg]).T
 
