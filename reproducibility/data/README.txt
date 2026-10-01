curated_dataset.csv            one row per compound: ChEMBL identifier, source DOI,
                               assay description, canonical SMILES, InChIKey,
                               IC50, pIC50, Bemis-Murcko scaffold, category
partitions.csv                 compound_index / partition membership for all five splits
partition_characterisation.csv similarity and scaffold-overlap statistics per split
representations.npz            every molecular representation matrix, keyed by name,
                               row order identical to curated_dataset.csv
repeated_partitions.csv        compound_index / role for the repeated 9:1 partitions of the
                               four resampleable designs (second revision)
primary_rule_replicates.csv    compound_index / role for the replicates of the primary scaffold-
                               and cluster-disjoint partitions under their own rule
repeated_partitions_all_folds.json  assignment of every compound to one of ten parts per design
repeated_partition_characterisation.csv  similarity and scaffold-overlap statistics per
                               repeated partition
