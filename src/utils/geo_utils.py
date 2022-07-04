import geopandas as gpd
from shapely.geometry import Point

def grow_point(series, square_size, epsg):
    # square_size = 50000 ## Square size in meters
    series = series.to_crs(epsg)
    polygon = series['geometry'].iloc[0].buffer(square_size/2, cap_style = 3)
    polygon_df = gpd.GeoDataFrame(index=[0], crs=epsg, geometry=[polygon])
    return polygon_df

def create_geodataframe(series):
    series['geometry'] = Point(series.longitude, series.latitude)
    series = series.to_frame().T
    try:
        zone = series.zone #hail
    except:
        zone = [item[:2] for item in list(series.mgrs.values)][0]  # non hail
    epsg = 32600 + int(zone)  ## epsg UTM North
    series = gpd.GeoDataFrame(series, crs='EPSG:4326')
    series = series.to_crs('EPSG:' + str(epsg))
    return series