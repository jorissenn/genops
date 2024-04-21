import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

def get_roads_from_street_block(roads, street_blocks, block_id):
    '''Returns the roads of a given street block.'''
    # extracting the street block and its geometry with the corresponding block_id
    street_block = street_blocks.copy()[street_blocks["block_id"] == block_id]
    street_block_geom = street_block.geometry.iloc[0]

    # extracting the street block boundary and transforming back to GeoDataFrame
    street_block_boundary = gpd.GeoDataFrame(geometry=street_block.boundary)

    # there might also be roads within the street block that are not part of the boundary
    roads_within_block = roads[roads.geometry.within(street_block_geom)]
    roads_within_block = roads_within_block.rename(columns={"geom": "geometry"})

    # concatenating the boundary and the interior roads of the street block
    roads_street_block = pd.concat([street_block_boundary, roads_within_block], ignore_index=True).reset_index(drop=True)
    
    return roads_street_block

def split_linestring(linestring):
    '''Split a LineString into individual segments'''
    points = list(linestring.coords)
    segments = [shapely.geometry.LineString([points[i], points[i+1]]) for i in range(len(points) - 1)]
    
    return segments

def dump_linestring(geometry):
    '''Process a LineString, splitting LineStrings and MultiLineStrings into segments'''
    if isinstance(geometry, shapely.geometry.LineString):
        return split_linestring(geometry)
    elif isinstance(geometry, shapely.geometry.MultiLineString):
        segments = []
        for linestring in geometry.geoms:
            segments.extend(split_linestring(linestring))
        return segments
    else:
        raise ValueError("Unsupported geometry type")

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