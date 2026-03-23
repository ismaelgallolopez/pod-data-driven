import os
from datetime import datetime, timedelta

# Central data directories on the aristarchos server
RES_DIR = "/home/ae4872/Resources"
MOD_DIR = "/home/ae4872/Models"
# Output directory to keep your workspace clean
OUT_DIR = "inputs"

def get_gps_week_day(date_obj):
    """Calculates the GPS week and day of week from a datetime object."""
    gps_epoch = datetime(1980, 1, 6)
    delta = date_obj - gps_epoch
    week = delta.days // 7
    day = delta.days % 7
    return f"{week:04d}{day}"

def generate_ghost_inps(target_date_obj):
    """Generates the three GHOST .inp files for a specific day."""
    year = target_date_obj.strftime('%Y')
    yy = target_date_obj.strftime('%y')
    month = target_date_obj.strftime('%m')
    day = target_date_obj.strftime('%d')
    doy = target_date_obj.strftime('%j')
    
    file_prefix = f"CMP_{yy}_{doy}"
    
    start_time = f"{year}/{month}/{day}  00:00:00.000"
    end_time = f"{year}/{month}/{day}  23:59:30.000"
    
    # Calculate GPS orbit file names for the day before, current day, and day after
    codx_prev = f"codx{get_gps_week_day(target_date_obj - timedelta(days=1))}.sp3"
    codx_curr = f"codx{get_gps_week_day(target_date_obj)}.sp3"
    codx_next = f"codx{get_gps_week_day(target_date_obj + timedelta(days=1))}.sp3"

    # 1. SPPLEO Template
    spp_content = f"""%
% {file_prefix}_2F_SPPLEO.inp
% Setup file for single point positioning of LEO satellites
%

% Observations
OBS_FILE            {RES_DIR}/Champ/{file_prefix}.rnx

OBS_START           {start_time}
OBS_END             {end_time}
OBS_STEP            0

% GPS orbit and clock data
GPS_ORB_FILE        {RES_DIR}/IGS/{codx_curr}

% GPS spacecraft description
ANTENNA_FILE        {MOD_DIR}/CMP_pod_est.atx
ANTENNA_FILE        {MOD_DIR}/igs05_1467_gnss.atx

% Processing options
DUAL_FREQUENCY
POSITION_ONLY

% Spacecraft parameters
SC_ID               L06               % SP3 spacecraft identifier

% Antenna offset in spacecraft body axes [m]
ANTENNA_NAME        CMP_POD_EST
ANTENNA_BORESIGHT   +0.000  +0.000  -1.000
ANTENNA_AZIMUTH     +1.000  +0.000  +0.000
ANTENNA_OFFSET      -1.4880   0.0000   -0.3928

% Data editing and weighting
EDIT_SNR            0.0
EDIT_ELEV           0.0
EDIT_CC             10.0
EDIT_SIG_PR         2.0
EDIT_SIG_RR         0.1
EDIT_PDOP           6.0
EDIT_NOBS           4

SIG_PR              1.5
SIG_RR              0.1

% Earth rotation parameters
LEAPS_FILE          {MOD_DIR}/leapsec.txt
EOP_FILE            {MOD_DIR}/igs96p02.erp
"""

    # 2. PosFit Template
    posfit_content = f"""%
% {file_prefix}_2F_PosFit.inp
% CHAMP setup file for dynamic ephemeris filtering
%

% Observations
NAV_FILE            {file_prefix}_2F_SPPLEO.sp3

NAV_START           {start_time}
NAV_END             {end_time}
NAV_STEP            1
EPH_STEP            10

% GPS orbit and clock data
GPS_ORB_FILE        {RES_DIR}/IGS/{codx_prev}
GPS_ORB_FILE        {RES_DIR}/IGS/{codx_curr}
GPS_ORB_FILE        {RES_DIR}/IGS/{codx_next}

% GPS spacecraft description
ANTENNA_FILE        {MOD_DIR}/igs05_1467_gnss.atx
ANTENNA_FILE        {MOD_DIR}/CMP_pod_est.atx

% Reference orbit
REF_ORBIT           {RES_DIR}/Champ/{file_prefix}.sp3

% Processing options
DUAL_FREQUENCY
ITERATIONS          3
PHASE_WINDUP_MODEL  CODE              % IGS, CODE, or NONE

% Ephemeris range
EPH_STEP            30                % [s]

% Spacecraft parameters
SC_MASS             500.0             % [kg]
SC_AREA             0.5               % [m^2]
SC_CR               10.0              % Radiation pressure coefficient
SC_CD               2.1               % Drag coefficient
SC_ID               L06               % SP3 spacecraft identifier

% Antenna offset in spacecraft body axes [m]
ANTENNA_NAME        CMP_POD_EST
ANTENNA_BORESIGHT   +0.000  +0.000  -1.000
ANTENNA_AZIMUTH     +1.000  +0.000  +0.000
ANTENNA_OFFSET      -1.4880   0.0000   -0.3928
ATTITUDE_FILE       {RES_DIR}/Champ/{file_prefix}.att

% Data editing 
EDIT_SNR            15.0
EDIT_ELEV           5.0               % [deg]
EDIT_SIG_PR         0.6               % [m]
EDIT_SIG_DCP        0.03              % [m]
EDIT_NOBS           2                 % presently unused !!

% Filter settings
FILTER_POS          100.0             % Position [m]
FILTER_VEL          100.0             % Velocity [m/s]
FILTER_CR           0.500             % Radiation pressure coefficient
FILTER_CD           1.0000            % Drag coefficient
FILTER_A_EMP_RAD    5.0e-9            % Empirical accelerations [m/s^2]
FILTER_A_EMP_TANG   50.0e-9           % in radial, tangential, normal
FILTER_A_EMP_NORM   50.0e-9           % direction
FILTER_A_EMP_TAU    600.0             % Emp. accel. time interval [s]

SIGMA_OBS           2.000
EDIT_OBS            10.00

% Gravity model
GRAVITY_MODEL_FILE  {MOD_DIR}/Grav_GGM02S.dat
GRAVITY_MODEL_N     100
GRAVITY_MODEL_M     100

% Ocean tide model
OCEAN_TIDE_FILE     {MOD_DIR}/OTIDES.TOPEX_4.0

% Solar-terrestrial flux
FLUX_FILE           {MOD_DIR}/flux.final

% Earth rotation parameters
LEAPS_FILE          {MOD_DIR}/leapsec.txt
EOP_FILE            {MOD_DIR}/igs96p02.erp
"""

    # 3. ODCP Template
    odcp_content = f"""%
% {file_prefix}_2F_ODCP.inp
% Batch orbit determination of CHAMP satellite (dual-frequency)
%

% Observations
OBS_FILE            {RES_DIR}/Champ/{file_prefix}.rnx

OBS_START           {start_time}
OBS_END             {end_time}
OBS_STEP            10

% GPS orbit and clock data
GPS_ORB_FILE        {RES_DIR}/IGS/{codx_prev}
GPS_ORB_FILE        {RES_DIR}/IGS/{codx_curr}
GPS_ORB_FILE        {RES_DIR}/IGS/{codx_next}

% GPS spacecraft description
ANTENNA_FILE        {MOD_DIR}/igs05_1467_gnss.atx
ANTENNA_FILE        {MOD_DIR}/CMP_pod_est.atx

% Reference orbit
REF_ORBIT           {file_prefix}_2F_PosFit.sp3

% Processing options
DUAL_FREQUENCY
ITERATIONS          5
PHASE_WINDUP_MODEL  IGS               % IGS, CODE, or NONE

% Ephemeris range
EPH_STEP            10                % [s]

% Spacecraft parameters
SC_MASS             500.0             % [kg]
SC_AREA             0.5               % [m^2]
SC_CR               10.0              % Radiation pressure coefficient
SC_CD               2.1               % Drag coefficient
SC_ID               L06               % SP3 spacecraft identifier

% Antenna offset in spacecraft body axes [m]
ANTENNA_NAME        CMP_POD_EST
ANTENNA_BORESIGHT   +0.000  +0.000  -1.000
ANTENNA_AZIMUTH     +1.000  +0.000  +0.000
ANTENNA_OFFSET      -1.4880   0.0000   -0.3928
ATTITUDE_FILE       {RES_DIR}/Champ/{file_prefix}.att

% Data editing 
EDIT_SNR            15.0
EDIT_ELEV           5.0               % [deg]
EDIT_SIG_PR         0.6               % [m]
EDIT_SIG_DCP        0.03              % [m]
EDIT_NOBS           2                 % presently unused !!

% Filter settings (a priori standard deviations, time scale)
FILTER_POS          100.0             % Position [m]
FILTER_VEL          100.0             % Velocity [m/s]
FILTER_CR           2.000             % Radiation pressure coefficient
FILTER_CD           1.0000            % Drag coefficient
FILTER_A_EMP_RAD    5.0e-9            % Empirical accelerations [m/s^2]
FILTER_A_EMP_TANG   30.0e-9           % in radial, tangential, normal
FILTER_A_EMP_NORM   10.0e-9           % direction
FILTER_A_EMP_TAU    600.0             % Emp. accel. time interval [s]
FILTER_BIAS         1.00              % Carrier phase bias sigma [m]

SIG_PR              0.5               % Pseudo range standard deviation [m]
SIG_CP              0.015             % Carrier Phase standard deviation [m]

% Gravity model
GRAVITY_MODEL_FILE  {MOD_DIR}/Grav_GGM02S.dat
GRAVITY_MODEL_N     100
GRAVITY_MODEL_M     100

% Ocean tide model
OCEAN_TIDE_FILE     {MOD_DIR}/OTIDES.TOPEX_4.0

% Solar-terrestrial flux
FLUX_FILE           {MOD_DIR}/flux.final

% Earth rotation parameters
LEAPS_FILE          {MOD_DIR}/leapsec.txt
EOP_FILE            {MOD_DIR}/igs96p02.erp
"""

    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Write files to the target directory
    with open(os.path.join(OUT_DIR, f"{file_prefix}_2F_SPPLEO.inp"), "w") as f:
        f.write(spp_content)
    with open(os.path.join(OUT_DIR, f"{file_prefix}_2F_PosFit.inp"), "w") as f:
        f.write(posfit_content)
    with open(os.path.join(OUT_DIR, f"{file_prefix}_2F_ODCP.inp"), "w") as f:
        f.write(odcp_content)

    print(f"Generated {file_prefix} inputs in ./{OUT_DIR}/")

if __name__ == "__main__":
    # Start date is April 1st, 2005 (DOY 091)
    start_date = datetime(2005, 4, 1)
    
    # Loop for 30 days up to DOY 120
    for i in range(30): 
        current_date = start_date + timedelta(days=i)
        generate_ghost_inps(current_date)