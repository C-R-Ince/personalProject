import pandas as pd
import numpy as np
from datetime import date
from premier_league import RankingTable 
import requests
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import matplotlib as plt
import argparse

startYear = 2015
curYear = startYear
currentDate = date.today()
currentYear = currentDate.year
i = 0
noYears = currentYear - startYear
clubs = []
leagueFin = pd.DataFrame()
totalFin = pd.DataFrame()

while i < (noYears-1):
    clubYear = []
    curYear = startYear + i
    nxtYear = curYear + 1
    season = str(curYear) + "-" + str(nxtYear)
    ranking = RankingTable(target_season=season).get_ranking_list()
    leagueTable = pd.DataFrame(ranking)
    leagueTable.columns = leagueTable.iloc[0]
    leagueTable = leagueTable.drop(index=0).reset_index(drop=True)
    leagueTable = leagueTable.iloc[:20]
    clubs = list(leagueTable['Team'])

    for club in clubs:
        if club == "Bournemouth":
            club = "afc-bournemouth"
        else:
            url = f"https://kinnaird-si-production.up.railway.app/api/clubs/{club}/dashboard-data?year={curYear}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            data = response.json()
            if "competitive_positions" not in data:
                #Skip those not in Kinnaird premier league will add championship later
                continue 
            # Competitive positions
            competitiveDf = pd.DataFrame(data["competitive_positions"]).T
            competitiveDf["Team"] = club
            competitiveDf["rankedExpense"] = competitiveDf.index
            mergeTable = pd.merge(leagueTable, competitiveDf, left_on = "Team", right_on = "Team")
            mergeTable = mergeTable[["Pos", "Team", "rankedExpense", "ranking"]]
            mergeTable = (
                mergeTable
                .pivot(index=["Pos", "Team"], 
                       columns="rankedExpense", 
                       values="ranking")
                .reset_index()
            )
            mergeTable.columns.name = None
            clubYear.append(mergeTable)
    leagueFin = pd.concat(clubYear, ignore_index=True)
    leagueFin["year"] = curYear
    leagueFin = leagueFin.drop("kinnaird_valuation", axis=1)
    totalFin = pd.concat([totalFin, leagueFin], ignore_index=True)
    i += 1
data = totalFin.dropna()
numeric_cols = ['net_debt','net_spend','player_signings','profit_loss_after_tax','staff_costs','total_income']

# Ensure numeric & handle NaNs
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
data['Pos'] = pd.to_numeric(data['Pos'], errors='coerce')
data['year'] = pd.to_numeric(data['year'], errors='coerce')

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--team", type=str, help=f'Specify team of interest. Please use quotation marks and choose from: {", ".join(clubs)}')
args = parser.parse_args()


results = []

# ML per season
all_years = sorted(data['year'].unique())
for test_year in all_years:
    train_data = data[data['year'] != test_year]
    test_data = data[data['year'] == test_year].copy()
    

    imputer_X = SimpleImputer(strategy='median')
    imputer_y = SimpleImputer(strategy='median')
    
    X_train = imputer_X.fit_transform(train_data[numeric_cols])
    y_train = imputer_y.fit_transform(train_data['Pos'].values.reshape(-1,1)).ravel()
    
    X_test = imputer_X.transform(test_data[numeric_cols])
    y_test = imputer_y.transform(test_data['Pos'].values.reshape(-1,1)).ravel()
    
    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
    
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).ravel()
    
    # Elastic Net
    en = ElasticNetCV(cv=3, l1_ratio=0.5, random_state=42)
    en.fit(X_train_scaled, y_train_scaled)
    
    y_pred_en_test = en.predict(X_test_scaled)
    
    # Input residulas for XGBoost
    residuals_train = y_train_scaled - en.predict(X_train_scaled)
    xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    xgb.fit(X_train_scaled, residuals_train)
    
    residuals_pred_test = xgb.predict(X_test_scaled)
    
    # Predict
    y_final_scaled = y_pred_en_test + residuals_pred_test
    y_final_test = scaler_y.inverse_transform(y_final_scaled.reshape(-1,1)).ravel()
    
    # Calculate Matrices
    mse_en = mean_squared_error(y_test_scaled, y_pred_en_test)
    mse_xgb = mean_squared_error(y_test_scaled - y_pred_en_test, residuals_pred_test)
    
    spearman_en = spearmanr(y_test_scaled, y_pred_en_test).correlation
    spearman_xgb = spearmanr(y_test_scaled - y_pred_en_test, residuals_pred_test).correlation # is XG overfitting creating near perfect results 
    
    print(f"Year {test_year}: ElasticNet MSE={mse_en:.4f}, Spearman={spearman_en:.4f}")
    print(f"Year {test_year}: XGBoost MSE={mse_xgb:.4f}, Spearman={spearman_xgb:.4f}")
    

    test_data['Predicted_Pos'] = y_final_test
    test_data['Residual'] = test_data['Pos'] - test_data['Predicted_Pos']
    results.append(test_data)

# Combine seasons
dfResults = pd.concat(results)

# Calculate rolling season values
dfResults.sort_values(['Team','year'], inplace=True)
dfResults['Residual_Rolling3'] = dfResults.groupby('Team')['Residual'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

# Efficiency percentage (currently calculates relative to club)
dfResults['Efficiency_Percentile'] = dfResults.groupby('year')['Residual'].rank(pct=True, ascending=False)

# Output
print(dfResults[['Team','year','Pos','Predicted_Pos','Residual','Residual_Rolling3','Efficiency_Percentile']])

# Get coefficients
# Sanity check 
#print(len(en.coef_))
#print(len(numeric_cols))
coef_df = pd.Series(en.coef_, index=numeric_cols)

# Plot coefficients
coef_df.sort_values().plot(kind="barh")
plt.title("Elastic Net Coefficients")
plt.xlabel("Coefficient Value")
plt.show()

# Check if team is specified
if args.team is not None:
    team_coef = dfResults[dfResults["Team"] == args.team]
    print(team_coef)

    fig, ax = plt.subplots(figsize=(10,6))

    ax.set_title(f"Team {args.team} Coefficients Over Years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Values")

    ax.invert_yaxis()

    ax.plot(team_coef["year"], team_coef["Pos"], label="Position")
    ax.plot(team_coef["year"], team_coef["Predicted_Pos"], label="Predicted Pos")

    ax.legend()
    plt.show()
