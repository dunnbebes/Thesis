reset;

# Sets
set SupplierID;
set FacilityID;
set ShipToID;
set ProductID;
set TransferFlows within {FacilityID, FacilityID};

# Parameters
param DemandInKG             {ShipToID,   ProductID}             >= 0 default 0;
## Facility
param StartingInventory      {FacilityID, ProductID}             >= 0 default 0;
param EndingInventory        {FacilityID, ProductID}             >= 0 default 0;
# param FacilityCapInKG        {FacilityID}                        >= 0 default 0;
# param FacilityCapInCBM       {FacilityID}                        >= 0 default 0;

# ## Conversion rate
# param CR_kg_to_CBM           {ProductID}                         >= 0 default 0;
# ## Cost
# param OrderingCost           {SupplierID, FacilityID}            >= 0 default 0;
# param TransportCost_transfer {FacilityID, FacilityID}            >= 0 default 0;
# param TransportCost_toShipTo {FacilityID, ShipToID}              >= 0 default 0;
# param RentingCost            {FacilityID}                        >= 0 default 0;
# param HandlingCost           {FacilityID}                        >= 0 default 0;
# param CustomerLoss                                               >= 0 default 0;


# Variables
var MetDemand     {ShipToID, ProductID}                    >= 0;
var UnmetDemand   {ShipToID, ProductID}                    >= 0;
var Purchases     {SupplierID, FacilityID, ProductID}      >= 0;
# var Transfers     {TransferFlows, ProductID}               >= 0; # tuple
var Transfers     {FacilityID, FacilityID, ProductID}      >= 0;
var TransferIn    {FacilityID, ProductID}                  >= 0;
var TransferOut   {FacilityID, ProductID}                  >= 0;
var Shipments     {FacilityID, ShipToID, ProductID}        >= 0;

# Objective
minimize Cost:
    sum {shipto in ShipToID, product in ProductID} UnmetDemand[shipto, product];

# Constraints
## Demand
subject to Demand_Balance {shipto in ShipToID, product in ProductID}:
    MetDemand[shipto, product] + UnmetDemand[shipto, product] = DemandInKG[shipto, product];
subject to ServeCustomer_Balance {shipto in ShipToID, product in ProductID}:
    MetDemand[shipto, product] = sum{fac in FacilityID} Shipments[fac, shipto, product];

## Flow Balancing
subject to Flow_Balance {fac in FacilityID, product in ProductID}:
    StartingInventory[fac, product] 
    + sum{sup in SupplierID} Purchases [sup, fac, product] 
    + TransferIn[fac, product]
                                                            = sum{shipto in ShipToID}    Shipments[fac, shipto, product] 
                                                            + TransferOut[fac, product] 
                                                            + EndingInventory[fac, product];
                                                            
subject to Flow_Transfer_In  {fac in FacilityID, product in ProductID}:
    TransferIn[fac, product]  = sum{(OriginFac, fac) in TransferFlows} Transfers[OriginFac, fac, product];
subject to Flow_Transfer_Out {fac in FacilityID, product in ProductID}:
    TransferOut[fac, product] = sum{(fac, DestinationFac) in TransferFlows} Transfers[fac, DestinationFac, product];
# subject to Flow_Transfer_In  {fac in FacilityID, product in ProductID}:
#     TransferIn[fac, product]  = sum{OriginFac in FacilityID} Transfers[OriginFac, fac, product];
# subject to Flow_Transfer_Out {fac in FacilityID, product in ProductID}:
#     TransferOut[fac, product] = sum{DestinationFac in FacilityID} Transfers[fac, DestinationFac, product];

## Facility Capacity
# subject to Facility_Capacity_inKG {fac in FacilityID}:
#     sum {product in ProductID} Inventory[fac, product] <= FacilityCapInKG[fac];
# subject to Facility_Capacity_inCBM {fac in FacilityID}:
#     sum {product in ProductID} Inventory[fac, product]*CR_kg_to_CBM[product] <= FacilityCapInCBM[fac];
