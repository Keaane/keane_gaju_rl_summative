"""All numbers the guide environment uses, in one place.

Every value is tagged:
  REAL       taken from a published source (see DATA_SOURCES.md for links)
  SYNTHETIC  no public data exists, so it is an assumption (reason given)
"""
from dataclasses import dataclass

# ---- Macro and formal-market numbers ---------------------------------------
INFLATION = 0.195        # REAL: NISR urban CPI, Oct 2026 (19.5% y/y; rural 31.3%)
TBILL_RETURN = 0.089     # REAL: 364-day T-bill, about 8.9% (BNR auction, NCBA Rwanda, Apr 2026)
TBILL_MIN = 100_000      # REAL: BNR minimum T-bill purchase, RWF 100,000
SACCO_RETURN = 0.07      # SYNTHETIC: SACCO deposit rates are not published (loan rates are 18-24%)
IDLE_RETURN = 0.0        # cash left idle earns nothing; inflation erodes it

# ---- Users -------------------------------------------------------------------
SAVINGS_MIN, SAVINGS_MAX = 100_000.0, 2_000_000.0   # SYNTHETIC: no public data on beginner savings
# Context (REAL): youth unemployment 18.8% vs adults 13.1% (NISR LFS, Q3 2026)


@dataclass(frozen=True)
class Opportunity:
    name: str
    area: int            # 0 agri, 1 reselling/importing, 2 services, 3 digital, 4 savings
    min_ticket: float    # RWF
    annual_return: float # nominal, if legitimate
    volatility: float    # annual
    time_needed: float   # 0 (none) to 1 (full-time) per week
    scam_prior: float    # SYNTHETIC chance that this offer is a scam in a given episode


AREAS = ["Agri & supplies", "Reselling & importing", "Services", "Digital work", "Savings"]

# Returns of the business opportunities are SYNTHETIC. They are set a few points above
# current inflation, because prices (and so resale margins) rise with it.
OPPORTUNITIES = [
    Opportunity("Agri input resale",          0,  50_000, 0.27, 0.20, 0.5, 0.00),
    Opportunity("Crop co-op farming", 0, 100_000, 0.32, 0.35, 1.0, 0.10),
    Opportunity("Local reselling",            1,  30_000, 0.30, 0.25, 0.8, 0.00),
    Opportunity("Import & resale",            1, 150_000, 0.45, 0.50, 0.8, 0.30),  # resale scams are common (e.g. fake "quinoa")
    Opportunity("Mobile money agent",         2,  80_000, 0.24, 0.10, 1.0, 0.05),  # ~196k agent SIMs, +10% a year: crowded
    Opportunity("Delivery / moto service",    2, 200_000, 0.35, 0.30, 1.0, 0.15),
    Opportunity("Freelance setup",            3,  30_000, 0.22, 0.30, 0.7, 0.00),
    Opportunity("Online trading group",       3,  20_000, 0.05, 0.60, 0.3, 0.60),  # BNR/CMA keep warning about these
    Opportunity("SACCO savings",              4,   5_000, SACCO_RETURN, 0.01, 0.1, 0.00),
    Opportunity("T-bills",                    4, TBILL_MIN, TBILL_RETURN, 0.01, 0.1, 0.00),
]
N_OPP = len(OPPORTUNITIES)
SACCO, TBILLS = 8, 9

# Array views used by the environment
import numpy as np
AREA_OF = np.array([o.area for o in OPPORTUNITIES])
MIN_TICKET = np.array([o.min_ticket for o in OPPORTUNITIES], dtype=float)
MU = np.array([o.annual_return for o in OPPORTUNITIES])
SIGMA = np.array([o.volatility for o in OPPORTUNITIES])
TIME_REQ = np.array([o.time_needed for o in OPPORTUNITIES])
SCAM_P = np.array([o.scam_prior for o in OPPORTUNITIES])
OPP_NAMES = [o.name for o in OPPORTUNITIES]
