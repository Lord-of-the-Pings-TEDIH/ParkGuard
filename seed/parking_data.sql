-- 10 Test Plate Numbers:
-- 1. B123ABC (ticket active)
-- 2. CJ05XYZ (ticket active)
-- 3. TM88DEF (ticket grace)
-- 4. IS07GHI (ticket expired)
-- 5. BV12KLM (ticket long expired)
-- 6. AA00AAA (ticket active)
-- 7. BB11BBB (ticket expired)
-- 8. CC22CCC (ticket grace)
-- 9. CT33NOP (sub active)
-- 10. PH44QRS (sub active)

-- Drop existing tables to align with test data requirements
DROP TABLE IF EXISTS parking_subscriptions CASCADE;
DROP TABLE IF EXISTS parking_tickets CASCADE;
DROP TABLE IF EXISTS parking_zones CASCADE;

-- Recreate parking_zones
CREATE TABLE parking_zones (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    code VARCHAR(50),
    grace_period_min INT
);

-- Recreate parking_tickets
CREATE TABLE parking_tickets (
    id SERIAL PRIMARY KEY,
    zone_id UUID REFERENCES parking_zones(id),
    plate_text VARCHAR(50),
    valid_from TIMESTAMP,
    valid_until TIMESTAMP
);

-- Recreate parking_subscriptions
CREATE TABLE parking_subscriptions (
    id SERIAL PRIMARY KEY,
    zone_id UUID REFERENCES parking_zones(id),
    plate_text VARCHAR(50),
    type VARCHAR(50),
    end_date TIMESTAMP
);

-- Insert parking zone
INSERT INTO parking_zones (id, name, code, grace_period_min) VALUES 
('00000000-0000-0000-0000-000000000001', 'Zona Test', 'ZT', 15);

-- Insert parking tickets
INSERT INTO parking_tickets (zone_id, plate_text, valid_from, valid_until) VALUES
('00000000-0000-0000-0000-000000000001', 'B123ABC', NOW() - INTERVAL '1 hour', NOW() + INTERVAL '2 hours'),
('00000000-0000-0000-0000-000000000001', 'CJ05XYZ', NOW(), NOW() + INTERVAL '45 minutes'),
('00000000-0000-0000-0000-000000000001', 'TM88DEF', NOW() - INTERVAL '1 hour', NOW() - INTERVAL '5 minutes'),
('00000000-0000-0000-0000-000000000001', 'IS07GHI', NOW() - INTERVAL '4 hours', NOW() - INTERVAL '3 hours'),
('00000000-0000-0000-0000-000000000001', 'BV12KLM', NOW() - INTERVAL '2 days', NOW() - INTERVAL '1 day'),
('00000000-0000-0000-0000-000000000001', 'AA00AAA', NOW() - INTERVAL '30 minutes', NOW() + INTERVAL '1 hour'),
('00000000-0000-0000-0000-000000000001', 'BB11BBB', NOW() - INTERVAL '5 hours', NOW() - INTERVAL '4 hours'),
('00000000-0000-0000-0000-000000000001', 'CC22CCC', NOW() - INTERVAL '2 hours', NOW() - INTERVAL '10 minutes');

-- Insert parking subscriptions
INSERT INTO parking_subscriptions (zone_id, plate_text, type, end_date) VALUES
('00000000-0000-0000-0000-000000000001', 'CT33NOP', 'monthly', CURRENT_DATE + INTERVAL '20 days'),
('00000000-0000-0000-0000-000000000001', 'PH44QRS', 'annual', CURRENT_DATE + INTERVAL '180 days');
