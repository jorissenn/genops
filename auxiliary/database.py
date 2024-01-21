import pandas as pd
import geopandas as gpd
import shapely.wkt

def read_table_from_db(engine, table_name, geom=False, geom_col="geom", columns_to_select=None):
    '''
    Reads data from a database table and returns a pandas DataFrame / GeoDataFrame.
    '''
    if columns_to_select:
        sql = f'''
        SELECT {', '.join(columns_to_select)}
        FROM public.{table_name};
        '''
    else:
        sql = f'''
        SELECT *
        FROM public.{table_name};
        '''

    if not geom:   
        df = pd.read_sql(sql, engine)
    
    else:
        df = gpd.read_postgis(sql, engine, geom_col=geom_col)
    
    return df

def read_table_from_db_multiple_geoms(engine, table_name, geom_cols, columns_to_select=None):
    '''Function that reads data from a PostGIS table containing multiple geometry columns to a geopandas dataframe'''
    assert len(geom_cols) >= 2
    
    # constructing a string to extract the additional geometry as WKT from the database
    geom_text = ""

    for geom in geom_cols[1:]:
        geom_text += f"ST_AsText({geom}) AS {geom}_text, "

    geom_text = geom_text[:-2] + " "
    
    # SQL query to extract the relevant information to a geopandas dataframe
    if columns_to_select:
        sql = f'''
        SELECT {', '.join(columns_to_select)},  
        {geom_text}
        FROM public.{table_name};
        '''
    else:
        sql = f'''
        SELECT *, {geom_text}
        FROM public.{table_name};
        '''        
    
    # setting the first geometry as the default geometry
    gdf = gpd.read_postgis(sql, engine, geom_col=geom_cols[0])
    
    for geom in geom_cols[1:]:
        # reconstructing the second geometry from the WKT
        gdf[geom] = gdf[f'{geom}_text'].apply(lambda x: shapely.wkt.loads(x))

        # dropping the unnecessary text column associated with the second geometry
        gdf.drop(f'{geom}_text', axis=1, inplace=True)

    return gdf