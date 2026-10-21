/**
 * Archivo JavaScript para la creación conceptual de colecciones en MongoDB.
 * Este script se ejecuta en el shell de MongoDB (mongosh).
 * 
 * Base de Datos: demand_db
 */

// 1. Conectarse a la base de datos (si no existe, la crea)
use demand_db;

// 2. Creación de la colección 'historical_sales'
// Propósito: Almacenar los registros históricos de ventas.
db.createCollection("historical_sales");

// 3. Creación de la colección 'active_forecasts'
// Propósito: Almacenar los resultados de los pronósticos de demanda.
db.createCollection("active_forecasts");

// 4. Creación de la colección 'user_profiles'
// Propósito: Almacenar información de usuarios y roles para la aplicación.
db.createCollection("user_profiles");

// Opcional: Insertar un documento de ejemplo en cada colección para inicializar
db.historical_sales.insertOne({
    producto: "Ejemplo Producto A",
    cliente: "Ejemplo Cliente 1",
    fecha: new Date(),
    pedido_piezas: 100
});

db.active_forecasts.insertOne({
    producto: "Ejemplo Producto A",
    cliente: "Ejemplo Cliente 1",
    cluster: 1,
    fecha_pronostico: new Date("2026-01-01"),
    pronostico_ensemble: 150.5,
    pronostico_min: 140.0,
    pronostico_max: 160.0
});

db.user_profiles.insertOne({
    username: "admin_user",
    role: "dbAdmin",
    email: "admin@example.com",
    created_at: new Date()
});

print("Colecciones creadas y documentos de ejemplo insertados en la base de datos 'demand_db'.");
