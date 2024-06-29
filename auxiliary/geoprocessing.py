import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

def dump_polygon_gdf_to_points(polygon_gdf):
    '''Given a GeoDataFrame consisting of polygons, dumps all polygon vertices and returns a GeoDataFrame with Points and the
    attributes of the original Polygon.'''
    # create an empty list to store data
    data = []

    # extract names of columns with geometry type
    geometry_columns = [col for col in polygon_gdf.columns if polygon_gdf[col].dtype == "geometry"]

    # extract name of active geometry column
    active_geom_column = polygon_gdf.geometry.name
    
    # iterate over each row in the GeoDataFrame
    for index, row in polygon_gdf.iterrows():
        geom = row[active_geom_column]

        if geom.geom_type == "Polygon":
            geom_list = [geom]
        elif geom.geom_type == "MultiPolygon":
            geom_list = list(geom.geoms)

        for poly in geom_list:
            # first and last vertex coordinates are contained twice and have to be removed
            geom_coords = list(dict.fromkeys(list(poly.exterior.coords)))

            # extract vertices from the geometry
            for x, y in geom_coords:
                data.append({"geometry": shapely.geometry.Point(x, y), **row.drop(geometry_columns)})

    points_gdf = gpd.GeoDataFrame(data, geometry="geometry")

    return points_gdf

def compute_intersections(gdf1, gdf2, exclude_self_intersections=False):
    '''
    Computes intersections between two GeoDataFrames and returns a numpy array with the indices of intersecting geometries 
    from both GeoDataFrames.
    '''
    # create spatial indices for both GeoDataFrames
    spatial_index1 = gdf1.sindex
    spatial_index2 = gdf2.sindex

    intersections = []

    # check for intersections between all pairs from gdf1 and gdf2
    for i, geom1 in enumerate(gdf1.geometry):
        # get the indices of the bounding box intersections
        possible_matches_index = list(spatial_index2.intersection(geom1.bounds))
        if possible_matches_index:
            # retrieve the potentially matching geometries from gdf2
            possible_matches = gdf2.iloc[possible_matches_index].reset_index(drop=False)
            precise_matches = possible_matches.geometry.intersects(geom1)

            # collect pairs of indices with actual intersections
            for match_index, is_intersect in precise_matches.items():
                if is_intersect:
                    original_index2 = possible_matches.at[match_index, 'index']
                    # optionally skip self-intersections
                    if not exclude_self_intersections or i != original_index2:
                        intersections.append([i, original_index2])

    # convert list to numpy array
    intersection_array = np.array(intersections)
    return intersection_array