-- Script SQL para la creación de la base de datos Collins_Demand_DB

-- Tabla: ventas_historicas
-- Contiene el registro histórico de pedidos y facturación.
CREATE TABLE IF NOT EXISTS "ventas_historicas" (
  "Producto_Descripcion" TEXT NOT NULL,
  "Cliente_Descripcion" TEXT NOT NULL,
  "Fecha" TIMESTAMP NOT NULL,
  "Pedido_Piezas" INTEGER NOT NULL,
  PRIMARY KEY ("Producto_Descripcion", "Cliente_Descripcion", "Fecha")
);

-- Tabla: pronosticos_activos
-- Contiene los resultados del modelo de pronóstico de demanda.
CREATE TABLE IF NOT EXISTS "pronosticos_activos" (
  "Producto_Descripcion" TEXT NOT NULL,
  "Cliente_Descripcion" TEXT NOT NULL,
  "Cluster" INTEGER,
  "Fecha_Pronostico" TIMESTAMP NOT NULL,
  "Pronostico_Ensemble_PedidoPiezas" REAL,
  "Pronostico_Min" REAL,
  "Pronostico_Max" REAL,
  PRIMARY KEY ("Producto_Descripcion", "Cliente_Descripcion", "Fecha_Pronostico")
);

-- Nota: Se asume que la combinación de Producto_Descripcion, Cliente_Descripcion y Fecha/Fecha_Pronostico
-- es la clave primaria para garantizar la unicidad de los registros.
