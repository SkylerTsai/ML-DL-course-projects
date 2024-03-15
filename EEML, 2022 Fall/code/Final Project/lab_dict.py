label_type = {
	# ID: ID
	'Id': None,
	# MSSubClass: Identifies the type of dwelling involved in the sale.
	'MSSubClass': ['20', '30', '40', '45', '50', '60', '70', '75', '80', '85', '90', '120', '150', '160', '180', '190'],
	# MSZoning: Identifies the general zoning classification of the sale.
	'MSZoning': ['A', 'C (all)', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'],
	# LotFrontage: Linear feet of street connected to property
	'LotFrontage': None,
	# LotArea: Lot size in square feet
	'LotArea': None,
	# Street: Type of road access to property
	'Street': ['Grvl', 'Pave'],
	# Alley: Type of alley access to property
	'Alley': ['Grvl', 'Pave', 'NA'],
	# LotShape: General shape of property
	'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
	# LandContour: Flatness of the property
	'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],
	# Utilities: Type of utilities available
	'Utilities': ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'],
	# LotConfig: Lot configuration
	'LotConfig': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
	# LandSlope: Slope of property
	'LandSlope': ['Gtl', 'Mod', 'Sev'],
	# Neighborhood: Physical locations within Ames city limits
	'Neighborhood': ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'],
	# Condition1: Proximity to various conditions
	'Condition1': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
	# Condition2: Proximity to various conditions (if more than one is present)
	'Condition2': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
	# BldgType: Type of dwelling, 'Twnhs' no in description
	'BldgType': ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'TwnhsI', 'Twnhs'],
	# HouseStyle: Style of dwelling
	'HouseStyle': ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'],
	# OverallQual: Rates the overall material and finish of the house
	'OverallQual': None,
	# OverallCond: Rates the overall condition of the house
	'OverallCond': None,
	# YearBuilt: Original construction date
	'YearBuilt': None,
	# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
	'YearRemodAdd': None,
	# RoofStyle: Type of roof
	'RoofStyle': ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'],
	# RoofMatl: Roof material
	'RoofMatl': ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
	# Exterior1st: Exterior covering on house
	'Exterior1st': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing'],
	# Exterior2nd: Exterior covering on house (if more than one material), 'Wd Shng', 'CmentBd', 'Brk Cmn' not in description
	'Exterior2nd': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing', 'Wd Shng', 'CmentBd', 'Brk Cmn'],
	# MasVnrType: Masonry veneer type
	'MasVnrType': ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone', 'NA'],
	# MasVnrArea: Masonry veneer area in square feet
	'MasVnrArea': None,
	# ExterQual: Evaluates the quality of the material on the exterior
	'ExterQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
	# ExterCond: Evaluates the present condition of the material on the exterior
	'ExterCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
	# Foundation: Type of foundation
	'Foundation': ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
	# BsmtQual: Evaluates the height of the basement
	'BsmtQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
	# BsmtCond: Evaluates the general condition of the basement
	'BsmtCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
	# BsmtExposure: Refers to walkout or garden level walls
	'BsmtExposure': ['Gd', 'Av', 'Mn', 'No', 'NA'],
	# BsmtFinType1: Rating of basement finished area
	'BsmtFinType1': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
	# BsmtFinSF1: Type 1 finished square feet
	'BsmtFinSF1': None,
	# BsmtFinType2: Rating of basement finished area (if multiple types)
	'BsmtFinType2': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
	# BsmtFinSF2: Type 2 finished square feet
	'BsmtFinSF2': None,
	# BsmtUnfSF: Unfinished square feet of basement area
	'BsmtUnfSF': None,
	# TotalBsmtSF: Total square feet of basement area
	'TotalBsmtSF': None,
	# Heating: Type of heating
	'Heating': ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'],
	# HeatingQC: Heating quality and condition
	'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
	# CentralAir: Central air conditioning
	'CentralAir': ['Y'],
	# Electrical: Electrical system
	'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix', 'NA'],
	# 1stFlrSF: First Floor square feet
	'1stFlrSF': None,
	# 2ndFlrSF: Second floor square feet
	'2ndFlrSF': None,
	# LowQualFinSF: Low quality finished square feet (all floors)
	'LowQualFinSF': None,
	# GrLivArea: Above grade (ground) living area square feet
	'GrLivArea': None,
	# BsmtFullBath: Basement full bathrooms
	'BsmtFullBath': None,
	# BsmtHalfBath: Basement half bathrooms
	'BsmtHalfBath': None,
	# FullBath: Full bathrooms above grade
	'FullBath': None,
	# HalfBath: Half baths above grade
	'HalfBath': None,
	# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
	'BedroomAbvGr': None,
	# Kitchen: Kitchens above grade
	'KitchenAbvGr': None,
	# KitchenQual: Kitchen quality
	'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
	# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
	'TotRmsAbvGrd': None,
	# Functional: Home functionality (Assume typical unless deductions are warranted)
	'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
	# Fireplaces: Number of fireplaces
	'Fireplaces': None,
	# FireplaceQu: Fireplace quality
	'FireplaceQu': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
	# GarageType: Garage location
	'GarageType': ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA'],
	# GarageYrBlt: Year garage was built
	'GarageYrBlt': None,
	# GarageFinish: Interior finish of the garage
	'GarageFinish': ['Fin', 'RFn', 'Unf', 'NA'],
	# GarageCars: Size of garage in car capacity
	'GarageCars': None,
	# GarageArea: Size of garage in square feet
	'GarageArea': None,
	# GarageQual: Garage quality
	'GarageQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
	# GarageCond: Garage condition
	'GarageCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
	# PavedDrive: Paved driveway
	'PavedDrive': ['Y', 'P', 'N'],
	# WoodDeckSF: Wood deck area in square feet
	'WoodDeckSF': None,
	# OpenPorchSF: Open porch area in square feet
	'OpenPorchSF': None,
	# EnclosedPorch: Enclosed porch area in square feet
	'EnclosedPorch': None,
	# 3SsnPorch: Three season porch area in square feet
	'3SsnPorch': None,
	# ScreenPorch: Screen porch area in square feet
	'ScreenPorch': None,
	# PoolArea: Pool area in square feet
	'PoolArea': None,
	# PoolQC: Pool quality
	'PoolQC': ['Ex', 'Gd', 'TA', 'Fa', 'NA'],
	# Fence: Fence quality
	'Fence': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA'],
	# MiscFeature: Miscellaneous feature not covered in other categories
	'MiscFeature': ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA'],
	# MiscVal: $Value of miscellaneous feature
	'MiscVal': None,
	# MoSold: Month Sold (MM)
	'MoSold': None,
	# YrSold: Year Sold (YYYY)
	'YrSold': None,
	# SaleType: Type of sale
	'SaleType': ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'],
	# SaleCondition: Condition of sale
	'SaleCondition': ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'],
	# SalePrice: Sale price
	'SalePrice': None,
}