from typing import *
from ..DataChain import *


class DataKeyLib_Raw:
	# data description
	# date:				List[int]	= [0, 0, 0]  # YYYY, MM, DD
	# time:				List[int]	= [0, 0, 0]  # HH, MM, SS
	# latitude:			float		= .0
	# longitude:		float		= .0
	# height:			int			= 0
	# turbulence:		int			= -1
	# pirep_turbulence:	int			= -1
	# pirep_icing:		int			= -1
	# filename_hs08:	str			= "HS_H08_"

	key_date:				DataKey = DataKey("Date")
	key_time:				DataKey = DataKey("Time")
	key_latitude:			DataKey = DataKey("Latitude")
	key_longitude:			DataKey = DataKey("Longitude")
	key_height:				DataKey = DataKey("Height")
	key_turbulence:			DataKey = DataKey("Turbulence")
	key_pirep_turbulence:	DataKey	= DataKey("PIREP_Turbulence")
	key_pirep_icing:		DataKey = DataKey("PIREP_Icing")
	key_filename_hs08:		DataKey = DataKey("Filename_HS08")
