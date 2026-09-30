# Data Architecture

## Business

- business_id
- business_type
- sector
- subsector
- location
- operating_calendar
- data_start_date
- lifecycle_status

## Sales

- transaction_id
- business_id
- timestamp
- product/category
- quantity
- unit_price
- gross_amount
- discount
- tax_amount
- return_amount
- net_sales
- payment_status

## Invoice

- invoice_id
- business_id
- invoice_date
- due_date
- customer_type
- invoice_amount
- outstanding_amount
- status

## Payment

- payment_id
- business_id
- timestamp
- amount
- direction
- payment_method
- linked_invoice_id

## Expense

- expense_id
- business_id
- expense_date
- category
- amount
- payment_date
- recurring_flag

## Inventory

- business_id
- product_id
- date
- opening_quantity
- purchases
- sales_quantity
- returns
- adjustments
- closing_quantity
- unit_cost

---

## Business State

The following must be distinguished:

- Operating
- Closed
- Stock-constrained
- Temporarily closed
- Data unavailable

Therefore:

**₹0 sales ≠ missing ≠ closed ≠ invalid**

---

## Financial Separation

Sales ≠ receivables ≠ cash.

Expense ≠ cash paid.

GST invoice data ≠ cash collection.

---

## Provenance

Important records should retain:

- source
- source record ID
- ingestion timestamp
- transformation version
- mapping version

---

## Forecast Passport

Each forecast should retain:

- forecast ID
- business ID
- target
- horizon
- expected value
- prediction interval
- generated timestamp
- data-through timestamp
- external-data vintage
- model version
- feature version
- evidence drivers
- assumptions
- data-quality status
