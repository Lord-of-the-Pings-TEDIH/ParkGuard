-- ParkGuard parking seed aligned with current SQLAlchemy schema.
-- Plates:
--   Active tickets:      B123ABC, CJ05XYZ, AA00AAA
--   Grace tickets:       TM88DEF, CC22CCC
--   Expired tickets:     IS07GHI, BV12KLM, BB11BBB
--   Active subscriptions: CT33NOP, PH44QRS

BEGIN;

WITH upsert_zone AS (
  INSERT INTO parking_zones (name, code, grace_period_min, is_active)
  VALUES ('Zona Test', 'ZT', 15, TRUE)
  ON CONFLICT (code)
  DO UPDATE SET
    name = EXCLUDED.name,
    grace_period_min = EXCLUDED.grace_period_min,
    is_active = EXCLUDED.is_active
  RETURNING id
)
SELECT 1;

DELETE FROM parking_tickets
WHERE plate_text IN ('B123ABC', 'CJ05XYZ', 'TM88DEF', 'IS07GHI', 'BV12KLM', 'AA00AAA', 'BB11BBB', 'CC22CCC');

DELETE FROM parking_subscriptions
WHERE plate_text IN ('CT33NOP', 'PH44QRS');

INSERT INTO parking_tickets (zone_id, plate_text, valid_from, valid_until, ticket_ref, payment_method) VALUES
((SELECT id FROM parking_zones WHERE code = 'ZT'), 'B123ABC', NOW() - INTERVAL '1 hour',    NOW() + INTERVAL '2 hours',   'TKT-ZT-001', 'SMS'),
((SELECT id FROM parking_zones WHERE code = 'ZT'), 'CJ05XYZ', NOW(),                          NOW() + INTERVAL '45 minutes','TKT-ZT-002', 'CARD'),
((SELECT id FROM parking_zones WHERE code = 'ZT'), 'TM88DEF', NOW() - INTERVAL '1 hour',    NOW() - INTERVAL '5 minutes','TKT-ZT-003', 'APP'),
((SELECT id FROM parking_zones WHERE code = 'ZT'), 'IS07GHI', NOW() - INTERVAL '4 hours',   NOW() - INTERVAL '3 hours',  'TKT-ZT-004', 'SMS'),
((SELECT id FROM parking_zones WHERE code = 'ZT'), 'BV12KLM', NOW() - INTERVAL '2 days',    NOW() - INTERVAL '1 day',    'TKT-ZT-005', 'CASH'),
((SELECT id FROM parking_zones WHERE code = 'ZT'), 'AA00AAA', NOW() - INTERVAL '30 minutes',NOW() + INTERVAL '1 hour',   'TKT-ZT-006', 'SMS'),
((SELECT id FROM parking_zones WHERE code = 'ZT'), 'BB11BBB', NOW() - INTERVAL '5 hours',   NOW() - INTERVAL '4 hours',  'TKT-ZT-007', 'CARD'),
((SELECT id FROM parking_zones WHERE code = 'ZT'), 'CC22CCC', NOW() - INTERVAL '2 hours',   NOW() - INTERVAL '10 minutes','TKT-ZT-008','APP');

INSERT INTO parking_subscriptions (
  zone_id, plate_text, subscription_type, start_date, end_date, holder_name, ref_number, is_active
) VALUES
((SELECT id FROM parking_zones WHERE code = 'ZT'), 'CT33NOP', 'MONTHLY', CURRENT_DATE - INTERVAL '10 days', CURRENT_DATE + INTERVAL '20 days', 'Test User 1', 'SUB-ZT-001', TRUE),
((SELECT id FROM parking_zones WHERE code = 'ZT'), 'PH44QRS', 'YEARLY',  CURRENT_DATE - INTERVAL '30 days', CURRENT_DATE + INTERVAL '180 days', 'Test User 2', 'SUB-ZT-002', TRUE);

COMMIT;
