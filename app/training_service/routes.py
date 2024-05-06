from fastapi import APIRouter
from training_service.local.image_classifier import routes as image_classifier_routes
from training_service.local.tabular.classification import routes as tabular_classifier_routes

router = APIRouter(prefix="/api/training_service",tags=["training_service"])

router.include_router(image_classifier_routes.router)
router.include_router(tabular_classifier_routes.router)


