# Predicting Gentrification Types from Housing Market Dynamics

Machine learning classification of gentrification processes at the census-section level across Spain, using quarterly housing price data from 2012-2022.

## Repository Structure

### `Trabajo fin de Grado Español/`
Original undergraduate thesis (TFG) in Spanish: *"La redistribucion del espacio despues de la crisis economica: Un estudio de la gentrificacion espanola"*. Includes the final PDF, presentation, and original data transformation code.

### `Published paper/`
English academic paper with all code for the machine learning pipeline:
- `paper.tex` / `paper.pdf` - The paper
- `src/` - Core modules (data loading, gentrification classification, feature engineering)
- `outputs/` - Experiment scripts and generated figures
- `config.py` - Central configuration
- `regenerate_figures.py` - Publication-quality figure generation

## Key Results

- **42% five-class accuracy** (2.1x the 20% chance baseline) predicting next-quarter gentrification type
- **0.77 AUC** on binary displacement-risk detection
- Systematic comparison of 11 algorithms across 4 families
- Census 2011 demographics and INE macro data add negligible predictive power

## Data

The Idealista housing price database is not included due to size and licensing. The pipeline expects the data at the path configured in `config.py`.

## Author

Eden Rochman Sharabi
