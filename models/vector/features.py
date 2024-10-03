# features are always specified in this order
feature_order = ("area", 
                 "perimeter", 
                 "convexity", 
                 "eri", 
                 "orientation_mbr", 
                 "wall_average", 
                 "voronoi_area", 
                 "impact_area", 
                 "x_coord", 
                 "y_coord")

# features for models determined through ablation study
important_features = {
    "HGNN elimination": [
        "area",
        "impact_area"
    ],
    "HGNN selection": [
        "area", 
        "perimeter", 
        "convexity", 
        "orientation_mbr", 
        "wall_average", 
        "voronoi_area", 
        "impact_area", 
        "x_coord", 
        "y_coord"
    ],
    "HGT elimination": [
        "area", 
        "perimeter", 
        "convexity", 
        "eri", 
        "orientation_mbr", 
        "wall_average", 
        "voronoi_area", 
        "impact_area", 
        "x_coord", 
        "y_coord"
    ],
    "HGT selection": [
        "area", 
        "perimeter", 
        "convexity", 
        "eri", 
        "orientation_mbr", 
        "wall_average", 
        "voronoi_area", 
        "impact_area", 
        "x_coord", 
        "y_coord"
    ]
}