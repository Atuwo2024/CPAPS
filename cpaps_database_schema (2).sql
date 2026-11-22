CREATE DATABASE IF NOT EXISTS cpaps_db;
USE cpaps_db;
CREATE TABLE locations (
    location_id     INT AUTO_INCREMENT PRIMARY KEY,
    county          VARCHAR(100) NOT NULL,
    sub_county      VARCHAR(100),
    ward            VARCHAR(100),
    street_address  VARCHAR(255),
    latitude        DECIMAL(10, 8),
    longitude       DECIMAL(11, 8),
    area_type       ENUM('Urban', 'Suburban', 'Rural') DEFAULT 'Urban',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE crime_types (
    crime_type_id   INT AUTO_INCREMENT PRIMARY KEY,
    category        VARCHAR(100) NOT NULL,
    subcategory     VARCHAR(100) NOT NULL,
    severity_level  ENUM('Low', 'Medium', 'High', 'Critical') NOT NULL,
    description     TEXT
);
CREATE TABLE suspects (
    suspect_id      INT AUTO_INCREMENT PRIMARY KEY,
    first_name      VARCHAR(100),
    last_name       VARCHAR(100),
    alias           VARCHAR(100),
    gender          ENUM('Male', 'Female', 'Unknown') DEFAULT 'Unknown',
    age_estimate    INT,
    nationality     VARCHAR(100) DEFAULT 'Kenyan',
    status          ENUM('At Large', 'Arrested', 'Convicted', 'Released', 'Unknown') DEFAULT 'Unknown',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE officers (
    officer_id      INT AUTO_INCREMENT PRIMARY KEY,
    badge_number    VARCHAR(50) UNIQUE NOT NULL,
    first_name      VARCHAR(100) NOT NULL,
    last_name       VARCHAR(100) NOT NULL,
    officer_rank    VARCHAR(100),
    station         VARCHAR(150),
    county          VARCHAR(100),
    phone           VARCHAR(20),
    email           VARCHAR(150),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE crime_incidents (
    incident_id         INT AUTO_INCREMENT PRIMARY KEY,
    case_number         VARCHAR(50) UNIQUE NOT NULL,
    crime_type_id       INT NOT NULL,
    location_id         INT NOT NULL,
    officer_id          INT NOT NULL,
    incident_date       DATE NOT NULL,
    incident_time       TIME,
    reported_date       DATE NOT NULL,
    description         TEXT,
    status              ENUM('Open', 'Under Investigation', 'Closed', 'Cold Case') DEFAULT 'Open',
    weather_condition   ENUM('Clear', 'Rainy', 'Cloudy', 'Night', 'Unknown') DEFAULT 'Unknown',
    property_loss_ksh   DECIMAL(15, 2) DEFAULT 0.00,
    injuries_reported   BOOLEAN DEFAULT FALSE,
    fatalities          INT DEFAULT 0,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (crime_type_id) REFERENCES crime_types(crime_type_id),
    FOREIGN KEY (location_id)   REFERENCES locations(location_id),
    FOREIGN KEY (officer_id)    REFERENCES officers(officer_id)
);
CREATE TABLE incident_suspects (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    incident_id     INT NOT NULL,
    suspect_id      INT NOT NULL,
    role            ENUM('Primary', 'Accomplice', 'Person of Interest') DEFAULT 'Person of Interest',
    arrested        BOOLEAN DEFAULT FALSE,
    arrest_date     DATE,
    FOREIGN KEY (incident_id) REFERENCES crime_incidents(incident_id),
    FOREIGN KEY (suspect_id)  REFERENCES suspects(suspect_id)
);
CREATE TABLE predictions (
    prediction_id       INT AUTO_INCREMENT PRIMARY KEY,
    location_id         INT NOT NULL,
    crime_type_id       INT NOT NULL,
    predicted_date      DATE NOT NULL,
    confidence_score    DECIMAL(5, 4),
    model_used          VARCHAR(100),
    risk_level          ENUM('Low', 'Medium', 'High', 'Critical') DEFAULT 'Medium',
    generated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verified            BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (location_id)   REFERENCES locations(location_id),
    FOREIGN KEY (crime_type_id) REFERENCES crime_types(crime_type_id)
);
CREATE TABLE hotspots (
    hotspot_id      INT AUTO_INCREMENT PRIMARY KEY,
    location_id     INT NOT NULL,
    crime_type_id   INT,
    incident_count  INT DEFAULT 0,
    risk_score      DECIMAL(5, 2),
    identified_date DATE,
    active          BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (location_id)   REFERENCES locations(location_id),
    FOREIGN KEY (crime_type_id) REFERENCES crime_types(crime_type_id)
);
INSERT INTO crime_types (category, subcategory, severity_level, description) VALUES
('Violent Crime',   'Murder',               'Critical', 'Unlawful killing of a person'),
('Violent Crime',   'Robbery with Violence','Critical', 'Theft using force or threat'),
('Violent Crime',   'Assault',              'High',     'Physical attack on a person'),
('Violent Crime',   'Rape/Sexual Assault',  'Critical', 'Non-consensual sexual acts'),
('Property Crime',  'Burglary',             'High',     'Breaking into premises to steal'),
('Property Crime',  'Theft',                'Medium',   'Taking property without violence'),
('Property Crime',  'Vehicle Theft',        'High',     'Stealing of motor vehicles'),
('Property Crime',  'Vandalism',            'Low',      'Deliberate damage to property'),
('Cyber Crime',     'Fraud/Scam',           'Medium',   'Online financial deception'),
('Cyber Crime',     'Identity Theft',       'High',     'Stealing personal information'),
('Drug Offences',   'Drug Trafficking',     'Critical', 'Illegal sale and supply of drugs'),
('Drug Offences',   'Drug Possession',      'Medium',   'Found in possession of narcotics'),
('Public Order',    'Disorderly Conduct',   'Low',      'Causing public disturbance'),
('Public Order',    'Illegal Gathering',    'Medium',   'Unlawful public assembly');

INSERT INTO locations (county, sub_county, ward, area_type, latitude, longitude) VALUES
('Nairobi',   'Westlands',      'Parklands',        'Urban',    -1.2635, 36.8030),
('Nairobi',   'Starehe',        'Ngara',            'Urban',    -1.2833, 36.8167),
('Nairobi',   'Embakasi East',  'Utawala',          'Suburban', -1.3000, 36.9167),
('Nairobi',   'Kibra',          'Laini Saba',       'Urban',    -1.3167, 36.7833),
('Mombasa',   'Mvita',          'Tudor',            'Urban',    -4.0435, 39.6682),
('Kisumu',    'Kisumu Central', 'Market Milimani',  'Urban',    -0.0917, 34.7680),
('Nakuru',    'Nakuru Town East','Biashara',         'Urban',    -0.3031, 36.0800),
('Machakos',  'Machakos Town',  'Muvuti',           'Suburban', -1.5177, 37.2634);

INSERT INTO officers (badge_number, first_name, last_name, officer_rank, station, county) VALUES
('NPS-001', 'James',   'Mwangi',   'Inspector',        'Westlands Police Station', 'Nairobi'),
('NPS-002', 'Grace',   'Achieng',  'Sergeant',         'Central Police Station',   'Nairobi'),
('NPS-003', 'Peter',   'Kamau',    'Constable',        'Kibra Police Post',        'Nairobi'),
('NPS-004', 'Fatuma',  'Ali',      'Detective',        'Mombasa Central',          'Mombasa'),
('NPS-005', 'Samuel',  'Otieno',   'Senior Sergeant',  'Kisumu Police Station',    'Kisumu');

INSERT INTO crime_incidents 
(case_number, crime_type_id, location_id, officer_id, incident_date, incident_time, reported_date, status, property_loss_ksh, injuries_reported, fatalities) 
VALUES
('NPS/2024/001', 2, 1, 1, '2024-01-05', '22:30:00', '2024-01-06', 'Closed',              15000.00, TRUE,  0),
('NPS/2024/002', 5, 2, 2, '2024-01-12', '03:15:00', '2024-01-12', 'Under Investigation', 45000.00, FALSE, 0),
('NPS/2024/003', 6, 3, 3, '2024-01-20', '14:00:00', '2024-01-20', 'Closed',              8000.00,  FALSE, 0),
('NPS/2024/004', 1, 4, 3, '2024-02-01', '23:00:00', '2024-02-02', 'Open',                0.00,     TRUE,  1),
('NPS/2024/005', 9, 1, 1, '2024-02-14', '10:00:00', '2024-02-14', 'Under Investigation', 200000.00,FALSE, 0),
('NPS/2024/006', 7, 5, 4, '2024-02-20', '01:30:00', '2024-02-21', 'Open',                350000.00,FALSE, 0),
('NPS/2024/007', 3, 2, 2, '2024-03-03', '18:45:00', '2024-03-03', 'Closed',              0.00,     TRUE,  0),
('NPS/2024/008', 11,6, 5, '2024-03-15', '20:00:00', '2024-03-16', 'Under Investigation', 0.00,     FALSE, 0),
('NPS/2024/009', 5, 7, 1, '2024-04-02', '02:00:00', '2024-04-02', 'Closed',              120000.00,FALSE, 0),
('NPS/2024/010', 2, 8, 2, '2024-04-18', '21:15:00', '2024-04-19', 'Open',                30000.00, TRUE,  0);

CREATE OR REPLACE VIEW v_crime_report AS
SELECT 
    ci.case_number,
    ci.incident_date,
    ci.incident_time,
    ct.category         AS crime_category,
    ct.subcategory      AS crime_type,
    ct.severity_level,
    l.county,
    l.sub_county,
    l.ward,
    l.area_type,
    l.latitude,
    l.longitude,
    CONCAT(o.officer_rank, ' ', o.first_name, ' ', o.last_name) AS reporting_officer,
    o.station,
    ci.status,
    ci.property_loss_ksh,
    ci.injuries_reported,
    ci.fatalities,
    DAYNAME(ci.incident_date)   AS day_of_week,
    MONTH(ci.incident_date)     AS month_number,
    YEAR(ci.incident_date)      AS year_number,
    HOUR(ci.incident_time)      AS hour_of_day
FROM crime_incidents ci
JOIN crime_types ct ON ci.crime_type_id = ct.crime_type_id
JOIN locations   l  ON ci.location_id   = l.location_id
JOIN officers    o  ON ci.officer_id    = o.officer_id;

CREATE OR REPLACE VIEW v_hotspot_summary AS
SELECT 
    l.county,
    l.sub_county,
    l.ward,
    l.latitude,
    l.longitude,
    COUNT(ci.incident_id)   AS total_incidents,
    SUM(ci.fatalities)      AS total_fatalities,
    SUM(ci.property_loss_ksh) AS total_loss_ksh
FROM crime_incidents ci
JOIN locations l ON ci.location_id = l.location_id
GROUP BY l.location_id, l.county, l.sub_county, l.ward, l.latitude, l.longitude
ORDER BY total_incidents DESC;

CREATE OR REPLACE VIEW v_monthly_trend AS
SELECT 
    YEAR(incident_date)     AS year,
    MONTH(incident_date)    AS month,
    ct.category,
    COUNT(*)                AS incident_count
FROM crime_incidents ci
JOIN crime_types ct ON ci.crime_type_id = ct.crime_type_id
GROUP BY YEAR(incident_date), MONTH(incident_date), ct.category
ORDER BY year, month;
