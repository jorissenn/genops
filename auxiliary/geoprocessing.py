import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

def get_sample_points(linestrings, dist):
    '''Given a set of linestrings, unions their geometry and samples a point every dist meters'''
    # union the roads of a street block to a single MultiLineString
    linestrings_union = shapely.ops.unary_union(linestrings.geometry)

    # sample points along the road every dist m
    sampled_points = [linestrings_union.interpolate(d) for d in range(0, int(linestrings_union.length), dist)]

    # convert to GeoDataFrame
    sampled_points_gdf = gpd.GeoDataFrame(geometry=sampled_points)
    
    # assign id according to the index
    sampled_points_gdf = sampled_points_gdf.assign(id=sampled_points_gdf.index+1)

    return sampled_points_gdf

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
        # first and last vertex coordinates are contained twice and have to be removed
        geom_coords = list(dict.fromkeys(list(geom.exterior.coords)))

        # extract vertices from the geometry
        for x, y in geom_coords:
            data.append({"geometry": shapely.geometry.Point(x, y), **row.drop(geometry_columns)})

    points_gdf = gpd.GeoDataFrame(data, geometry="geometry")

    return points_gdf

def construct_voronoi_polygons(buildings, street_blocks, block_id):
    '''Constructs Voronoi polygons with the individual vertices of the buildings inside of a street block with given block_id'''
    # extract centroids within street block block_id
    buildings_block = buildings.copy()[buildings["block_id"] == block_id]

    # dump all convex hull vertices to a new DataFrame
    building_points_gdf = dump_polygon_gdf_to_points(buildings_block)
    building_points = building_points_gdf.geometry
    
    # extract geometry of street block block_id
    street_block = street_blocks.copy()[street_blocks["block_id"] == block_id]
    street_block_buffered = street_block.buffer(100)

    # transforming Points to MultiPoints
    buildings_point_multipoint = shapely.geometry.MultiPoint([building_point for building_point in building_points])

    # constructing Voronoi polygons and transforming to GeoDataFrame
    voronoi_polygons = shapely.voronoi_polygons(buildings_point_multipoint, extend_to=street_block_buffered.geometry.iloc[0])
    voronoi_polygons_gdf = gpd.GeoDataFrame(geometry=list(voronoi_polygons.geoms))
    voronoi_polygons_gdf = voronoi_polygons_gdf.set_crs("epsg:2056")

    # assign uuid of original buildings to Voronoi polygons
    voronoi_polygons_gdf = gpd.sjoin(voronoi_polygons_gdf, building_points_gdf, how="inner", predicate="intersects")
    voronoi_polygons_gdf = voronoi_polygons_gdf.drop(labels="index_right", axis=1)

    return voronoi_polygons_gdf

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