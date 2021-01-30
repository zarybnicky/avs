/**
 * @file    cached_mesh_builder.cpp
 * @brief   Parallel Marching Cubes implementation using pre-computed field
 * @author  Jakub Zárybnický <xzaryb00@stud.fit.vutbr.cz>
 * @date    20.12.2020
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <omp.h>

#include "cached_mesh_builder.h"

CachedMeshBuilder::CachedMeshBuilder(unsigned gridEdgeSize)
  : BaseMeshBuilder(gridEdgeSize, "Cached") { }

unsigned CachedMeshBuilder::marchCubes(const ParametricScalarField &field) {
  size_t cacheSize = (mGridSize + 1) * (mGridSize + 1) * (mGridSize + 1);
  mCache.resize(cacheSize);
  #pragma omp parallel for
  for (size_t i = 0; i < cacheSize; i++) {
    Vec3_t<float> pos((i % (mGridSize + 1)) * mGridResolution,
                      ((i / (mGridSize + 1)) % (mGridSize + 1)) * mGridResolution,
                      (i / (mGridSize + 1) / (mGridSize + 1)) * mGridResolution);
    float value = std::numeric_limits<float>::max();
    for (auto p : field.getPoints()) {
      float distanceSquared
        = (pos.x - p.x) * (pos.x - p.x)
        + (pos.y - p.y) * (pos.y - p.y)
        + (pos.z - p.z) * (pos.z - p.z);
      value = std::min(value, distanceSquared);
    }
    mCache[i] = sqrt(value);
  }

  //Reused from LoopMeshBuilder
  size_t totalCubesCount = mGridSize * mGridSize * mGridSize;
  unsigned totalTriangles = 0;
  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    #pragma omp single
    mTriangles = new std::vector<std::vector<Triangle_t> >(nthreads);
    #pragma omp for
    for (size_t i = 0; i < totalCubesCount; ++i) {
      Vec3_t<float> cubeOffset(i % mGridSize, (i / mGridSize) % mGridSize, i / mGridSize / mGridSize);
      #pragma omp atomic
      totalTriangles += buildCube(cubeOffset, field);
    }
  }
  for (auto row : *mTriangles) {
    outVector.insert(outVector.end(), row.begin(), row.end());
  }
  return totalTriangles;
}

float CachedMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field) {
  unsigned idx
    = floor(pos.x / mGridResolution + 0.5)
    + floor(pos.y / mGridResolution * (mGridSize + 1) + 0.5)
    + floor(pos.z / mGridResolution * (mGridSize + 1) * (mGridSize + 1) + 0.5);
  return mCache[idx];
}

void CachedMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle) {
  (*mTriangles)[omp_get_thread_num()].push_back(triangle);
}

const BaseMeshBuilder::Triangle_t *CachedMeshBuilder::getTrianglesArray() const {
  return outVector.data();
}
