reset;

# Sets
set SupplierID;
set FacilityID;
set ShipToID;
set ProductID;

# Parameters
param DemandInKG {ShipToID, ProductID} >= 0 default 0;
param SupplierCapInKG {SupplierID, ProductID} >= 0 default 0;
param FacilityCapInKG {FacilityID} >= 0 default 0;

# Variables
var MetDemand {ShipToID, ProductID} >= 0;
var UnmetDemand {ShipToID, ProductID} >= 0;
var Purchase {SupplierID, FacilityID, ProductID} >= 0;
var InventoryFac1 {FacilityID, ProductID} >= 0;
var InventoryFac2 {FacilityID, ProductID} >= 0;
var TransferIn {originFac in FacilityID, destFac in FacilityID, ProductID} >= 0;
var TransferOut {originFac in FacilityID, destFac in FacilityID, ProductID} >= 0;
var ServeCustomer {FacilityID, ShipToID, ProductID} >= 0;

# Objective
minimize UnmetDemand_Obj:
    sum {cus in ShipToID, product in ProductID} UnmetDemand[cus, product];

# Constraints
## Demand
subject to Demand_Balance {cus in ShipToID, product in ProductID}:
    MetDemand[cus, product] + UnmetDemand[cus, product] = DemandInKG[cus, product];

subject to ServeCustomer_Balance {cus in ShipToID, product in ProductID}:
    MetDemand[cus, product] = sum {fac in FacilityID} ServeCustomer[fac, cus, product];

## Supply
subject to Supply_Balance {sup in SupplierID, product in ProductID}:
    sum {fac in FacilityID} Purchase[sup, fac, product] <= SupplierCapInKG[sup, product];

## Flow balancing
subject to Flow_Balance_Purchase {fac in FacilityID, product in ProductID}:
    InventoryFac1[fac, product] = sum {sup in SupplierID} Purchase[sup, fac, product];

subject to Flow_Balance_Inventory {fac in FacilityID, product in ProductID}:
    InventoryFac2[fac, product] >= sum {cus in ShipToID} ServeCustomer[fac, cus, product];

subject to Flow_Balance_Transfers {fac in FacilityID, product in ProductID}:
    InventoryFac2[fac, product] = InventoryFac1[fac, product]
                                   - sum {destinationFac in FacilityID} TransferOut[fac, destinationFac, product]
                                   + sum {originFac in FacilityID} TransferIn[originFac, fac, product];

## Facility Capacity
subject to Facility_Capacity_1 {fac in FacilityID}:
    sum {product in ProductID} InventoryFac1[fac, product] <= FacilityCapInKG[fac];

subject to Facility_Capacity_2 {fac in FacilityID}:
    sum {product in ProductID} InventoryFac2[fac, product] <= FacilityCapInKG[fac];
