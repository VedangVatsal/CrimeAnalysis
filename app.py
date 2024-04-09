import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_from_directory
from mpu import haversine_distance

app = Flask(__name__)

# Time Features
class FeaturizeTime:
    def __init__(self, X_q):
        self.X_q = X_q
    
    def extract_date(self, time):
        return time.split(' ')[0]

    def extract_year(self, date):
        return int(date.split('-')[0])

    def extract_month(self, date):
        return int(date.split('-')[1])

    def extract_day(self, date):
        return int(date.split('-')[2])

    def extract_hour(self, time):
        date, hms = time.split(' ')
        return int(hms.split(':')[0])

    def extract_minute(self, time):
        date, hms = time.split(' ')
        return int(hms.split(':')[1])

    def extract_season(self, month):
        if month in [4, 5, 6]:
            return 'summer'
        elif month in [7, 8, 9]:
            return 'rainy'
        elif month in [10, 11, 12]:
            return 'winter'
        return 'spring'

    def extract_hour_type(self, hour):
        if (hour >= 4) and (hour < 12):
            return 'morning'
        elif (hour >= 12) and (hour < 15):
            return 'noon'
        elif (hour >= 15) and (hour < 18):
            return 'evening'
        elif (hour >= 18) and (hour < 22):
            return 'night'
        return 'mid-night'

    def extract_time_period(self, hour):
        if hour in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            return 'am'
        return 'pm'
    
    def extract_address_type(self, addr):
        if ' / ' in addr:
            return 'Cross'
        addr_sep = addr.split(' ')
        addr_type = addr_sep[-1]
        return addr_type
    
    def featurize_time(self):
        time_val = self.X_q[0]
        addr_val = self.X_q[3]

        date = self.extract_date(time=time_val)
        year = self.extract_year(date=date)
        month = self.extract_month(date=date)
        day = self.extract_day(date=date)
        hour = self.extract_hour(time=time_val)
        minute = self.extract_minute(time=time_val)
        season = self.extract_season(month=month)
        hour_type = self.extract_hour_type(hour=hour)
        time_period = self.extract_time_period(hour=hour)
        addr_type = self.extract_address_type(addr=addr_val)
        
        v = pd.DataFrame()
        
        prev_vals = list(self.X_q)
        fe_vals = [date, year, month, day, hour, minute, season, hour_type, time_period, addr_type]
        v['vals'] = prev_vals + fe_vals

        prev_columns = ['time', 'weekday', 'police_dept', 'address', 'longitude', 'latitude']
        fe_columns = ['date', 'year', 'month', 'day', 'hour', 'minute', 'season', 'hour_type', 'time_period', 'addr_type']
        columns = prev_columns + fe_columns

        tdf = v.T
        tdf.columns = columns
        
        return tdf

# Categorical Features
class OneHotEncoding:
    def __init__(self, X_q):
        self.X_q = X_q

        self.bays = ['Bayview', 'Central', 'Ingleside', 'Mission', 'Northern', 
                     'Park', 'Richmond', 'Southern', 'Taraval', 'Tenderloin']
        self.days = ['Friday', 'Monday', 'Saturday', 'Sunday', 
                     'Thursday', 'Tuesday', 'Wednesday']
        self.ap = ['Am', 'Pm']
        self.seasons = ['Rainy', 'Spring', 'Summer', 'Winter']
        self.ht = ['Evening', 'Mid-Night', 'Morning', 'Night', 'Noon']
        self.streets = ['/', 'Al', 'Av', 'Bl', 'Bufano', 'Cr', 'Cross', 'Ct', 'Dr', 'Ex', 'Ferlinghetti',
                        'Hwy', 'Hy', 'I-80', 'Ln', 'Mar', 'Palms', 'Park', 'Pl', 'Pz', 'Rd', 
                        'Rw', 'St', 'Stwy', 'Ter', 'Tr', 'Way', 'Wk', 'Wy']
    
    def encode_ohe(self, val, val_types):
        res = [0] * len(val_types)
        val_index = val_types.index(val)
        res[val_index] = 1
        return res
    
    def get_dummies(self):
        X_q = self.X_q.values[0]
        
        bay_area = X_q[2].title()
        if bay_area == 'Bayview':
            fbay = [1] + ([0] * 9)
        elif bay_area == 'Central':
            fbay = [0] + [1] + ([0] * 8)
        elif bay_area == 'Ingleside':
            fbay = [0, 0] + [1] + ([0] * 7)
        elif bay_area == 'Mission':
            fbay = [0, 0, 0] + [1] + ([0] * 6)
        elif bay_area == 'Northern':
            fbay = ([0] * 4) + [1] + ([0] * 5)
        elif bay_area == 'Park':
            fbay = ([0] * 5) + [1] + ([0] * 4)
        elif bay_area == 'Richmond':
            fbay = ([0] * 6) + [1] + [0, 0, 0]
        elif bay_area == 'Southern':
            fbay = ([0] * 7) + [1] + [0, 0]
        elif bay_area == 'Taraval':
            fbay = ([0] * 8) + [1] + [0]
        elif bay_area == 'Tenderloin':
            fbay = ([0] * 9) + [1]
        else:
            fbay = [0] * 10
        
        if X_q[1] == 'Friday':
            fday = [1] + ([0] * 6)
        elif X_q[1] == 'Monday':
            fday = [0] + [1] + ([0] * 5)
        elif X_q[1] == 'Saturday':
            fday = [0, 0] + [1] + ([0] * 4)
        elif X_q[1] == 'Sunday':
            fday = ([0] * 3) + [1] + ([0] * 3)
        elif X_q[1] == 'Thursday':
            fday = ([0] * 4) + [1] + [0, 0]
        elif X_q[1] == 'Tuesday':
            fday = ([0] * 5) + [1] + [0]
        elif X_q[1] == 'Wednesday':
            fday = ([0] * 6) + [1]
        else:
            fday = [0] * 7
        
        if X_q[-2] == 'Am':
            f_tp = [1, 0]
        elif X_q[-2] == 'Pm':
            f_tp = [0, 1]
        else:
            f_tp = [0, 0]
        
        if X_q[-4] == 'Rainy':
            fseason = [1, 0, 0, 0]
        elif X_q[-4] == 'Spring':
            fseason = [0, 1, 0, 0]
        elif X_q[-4] == 'Summer':
            fseason = [0, 0, 1, 0]
        elif X_q[-4] == 'Winter':
            fseason = [0, 0, 0, 1]
        else:
            fseason = [0, 0, 0, 0]
        
        if X_q[-3] == 'Evening':
            f_ht = [1, 0, 0, 0, 0]
        elif X_q[-3] == 'Mid-Night':
            f_ht = [0, 1, 0, 0, 0]
        elif X_q[-3] == 'Morning':
            f_ht = [0, 0, 1, 0, 0]
        elif X_q[-3] == 'Night':
            f_ht = [0, 0, 0, 1, 0]
        elif X_q[-3] == 'Noon':
            f_ht = [0, 0, 0, 0, 1]
        else:
            f_ht = [0, 0, 0, 0, 0]
        
        for i in self.streets:
            if X_q[-1] == i:
                f_st = self.encode_ohe(i, self.streets)
                break
            else:
                continue
        
        X_new = list(X_q) + fseason + f_ht + f_st + fbay + f_tp + fday
        columns = list(self.X_q.columns) + self.seasons + self.ht + self.streets + self.bays + self.ap + self.days
        v = pd.DataFrame()
        v['vals'] = X_new
        tdf = v.T
        tdf.columns = columns

        tdf = tdf.drop(columns=['time', 'weekday', 'police_dept', 'date', 'season', 'hour_type', 'time_period', 'addr_type'], axis=1)

        return tdf

# Spatial Distance Features
class SpatialDistanceFeatures:
    def __init__(self, X_q):
        self.X_q = X_q
        self.sf_pstations_tourists = {
            "sfpd": [37.7725, -122.3894],
            "ingleside": [37.7247, -122.4463],
            "central": [37.7986, -122.4101],
            "northern": [37.7802, -122.4324],
            "mission": [37.7628, -122.4220],
            "tenderloin": [37.7838, -122.4129],
            "taraval": [37.7437, -122.4815],
            "sfpd park": [37.7678, -122.4552],
            "bayview": [37.7298, -122.3977],
            "kma438 sfpd": [37.7725, -122.3894],
            "richmond": [37.7801, -122.4644],
            "police commission": [37.7725, -122.3894],
            "juvenile": [37.7632, -122.4220],
            "southern": [37.6556, -122.4366],
            "sfpd pistol range": [37.7200, -122.4996],
            "sfpd public affairs": [37.7754, -122.4039],
            "broadmoor": [37.6927, -122.4748],
            "napa wine country": [38.2975, -122.2869],
            "sonoma wine country": [38.2919, -122.4580],
            "muir woods": [37.8970, -122.5811],
            "golden gate": [37.8199, -122.4783],
            "yosemite national park": [37.865101, -119.538330],
        }
    
    def get_distance(self, ij):
        i = ij[0]
        j = ij[1]
        distance = haversine_distance(origin=i, destination=j)
        return distance

    def extract_spatial_distance_feature(self):
        X_q = self.X_q.values[0]
        lat_val = X_q[2]
        lon_val = X_q[1]
        
        origin = [lat_val, lon_val]
        pnames = list(self.sf_pstations_tourists.keys())
        pcoords = list(self.sf_pstations_tourists.values())

        pdists = []
        for pc in pcoords:
            dist = self.get_distance(ij=[origin, pc])
            pdists.append(dist)
        
        v = pd.DataFrame()
        v['vals'] = pdists
        tdf = v.T
        tdf.columns = pnames

        return pd.concat(objs=[self.X_q, tdf], axis=1)

# LatLong Features
class LatLongFeatures:
    def __init__(self, X_q):
        self.X_q = X_q
    
    def lat_lon_sum(self, ll):
        lat = ll[0]
        lon = ll[1]
        return lat + lon

    def lat_lon_diff(self, ll):
        lat = ll[0]
        lon = ll[1]
        return lon - lat

    def lat_lon_sum_square(self, ll):
        lat = ll[0]
        lon = ll[1]
        return (lat + lon) ** 2

    def lat_lon_diff_square(self, ll):
        lat = ll[0]
        lon = ll[1]
        return (lat - lon) ** 2

    def lat_lon_sum_sqrt(self, ll):
        lat = ll[0]
        lon = ll[1]
        return (lat**2 + lon**2) ** (1 / 2)

    def lat_lon_diff_sqrt(self, ll):
        lat = ll[0]
        lon = ll[1]
        return (lon**2 - lat**2) ** (1 / 2)
    
    def extract_lat_lon_features(self):
        X_q = self.X_q.values[0]
        
        lat_val = X_q[2]
        lon_val = X_q[1]
        ll = [lat_val, lon_val]

        columns = ['lat_lon_sum', 'lat_lon_diff', 'lat_lon_sum_square', 
                   'lat_lon_diff_square', 'lat_lon_sum_sqrt', 'lat_lon_diff_sqrt']
        vals = [self.lat_lon_sum(ll), self.lat_lon_diff(ll), self.lat_lon_sum_square(ll), 
                self.lat_lon_diff_square(ll), self.lat_lon_sum_sqrt(ll), self.lat_lon_diff_sqrt(ll)]

        v = pd.DataFrame()
        v['vals'] = vals
        tdf = v.T
        tdf.columns = columns

        return pd.concat(objs=[self.X_q, tdf], axis=1)

# Address (BoW & TfIDF) Features
class AddressFeatures:
    def __init__(self, X_q):
        self.X_q = X_q
        
        self.best_tfidf_columns = [17, 236, 328, 421, 718, 869, 940, 1023, 1078, 1163, 1178, 
                                   1180, 1392, 1466, 1500, 1550, 1582, 1817, 1854, 1971]
    
    def extract_tfidf(self, address):
        model_name = 'vect_tfidf_address.pkl'
        vect = pickle.load(open('models/' + model_name, "rb"))
        f_addr = vect.transform(raw_documents=[address])
        f_addr = f_addr.toarray()[:, self.best_tfidf_columns]
        return f_addr[0]
    
    def extract_addr_features(self):
        X_q = self.X_q.values[0]
        address = X_q[0]
        tfidf_f = self.extract_tfidf(address=address)

        columns = self.best_tfidf_columns
        v = pd.DataFrame()
        v['vals'] = list(tfidf_f)
        tdf = v.T
        tdf.columns = columns

        tdf = pd.concat(objs=[self.X_q, tdf], axis=1)
        tdf = tdf.drop(columns=['address'], axis=1)
        
        return tdf

# Make Predictions
labels = [
    'ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY',
    'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC',
    'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION', 'FAMILY OFFENSES',
    'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING',
    'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON',
    'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT',
    'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY',
    'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE',
    'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS',
    'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS'
]

# Single Point Predictor
class CrimeClassifier:
    def __init__(self, X_q):
        # Feature engineering
        ft = FeaturizeTime(X_q=X_q)
        X_q = ft.featurize_time()

        fc = OneHotEncoding(X_q=X_q)
        X_q = fc.get_dummies()

        fsd = SpatialDistanceFeatures(X_q=X_q)
        X_q = fsd.extract_spatial_distance_feature()

        fll = LatLongFeatures(X_q=X_q)
        X_q = fll.extract_lat_lon_features()

        fa = AddressFeatures(X_q=X_q)
        X_q = fa.extract_addr_features()

        # Handle single row case
        if isinstance(X_q.shape, tuple) and (len(X_q.shape) == 1):
            X_q = X_q.reshape(1, -1)

        # Load scaler
        scaler = pickle.load(open('models/scaler.pkl', "rb"))
        X_q.columns = X_q.columns.astype(str)
        self.X_q = scaler.transform(X_q)

    def predict(self, model_name='xg_boost', labels=labels):
        model_path = 'models/'
        if model_name == 'decision_tree':
            model_path += 'decision_tree_classifier.pkl'
        elif model_name == 'random_forest':
            model_path += 'random_forest_classifier.pkl'
        elif model_name == 'log_reg':
            model_path += 'log_reg_classifier.pkl'
        else:
            model_path += 'xgboost_multi_classifier.pkl'

        model = pickle.load(open(model_path, 'rb'))
        probas = model.predict_proba(self.X_q)
        max_prob = np.argmax(probas)
        category = labels[max_prob]

        return category

# Flask Routes
@app.route('/')
def index():
    return render_template('sf.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from request.form
    time = request.form.get('time')
    weekday = request.form.get('weekday')
    bayarea = request.form.get('Police District')
    address = request.form.get('address')
    longitude = request.form.get('longitude')
    latitude = request.form.get('latitude')
    
    
    if None in [time, weekday, bayarea, address, longitude, latitude]:
        return render_template('sf.html', error_message="All fields are required!")

    # Create X_q array
    X_q = [time, weekday, bayarea, address, longitude, latitude]

    # Make prediction
    cc = CrimeClassifier(X_q=X_q)
    prediction = cc.predict()

    # Update the template with the prediction
    return render_template('sf.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
