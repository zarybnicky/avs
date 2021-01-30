/**
 * @file    loop_mesh_builder.cpp
 * @brief   Parallel Marching Cubes implementation using OpenMP loops
 * @author  Jakub Zárybnický <xzaryb00@stud.fit.vutbr.cz>
 * @date    20.12.2020
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <omp.h>

#include "loop_mesh_builder.h"

LoopMeshBuilder::LoopMeshBuilder(unsigned gridEdgeSize)
  : BaseMeshBuilder(gridEdgeSize, "OpenMP Loop") { }

unsigned LoopMeshBuilder::marchCubes(const ParametricScalarField &field) {
  size_t totalCubesCount = mGridSize * mGridSize * mGridSize;
  unsigned totalTriangles = 0;
  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    #pragma omp single
    mTriangles = new std::vector<std::vector<Triangle_t> >(nthreads);
    #pragma omp for schedule(dynamic, 16)
    for (size_t i = 0; i < totalCubesCount; ++i) {
      Vec3_t<float> cubeOffset(i % mGridSize, (i / mGridSize) % mGridSize, i / (mGridSize * mGridSize));
      #pragma omp atomic
      totalTriangles += buildCube(cubeOffset, field);
    }
  }
  for (auto row : *mTriangles) {
    outVector.insert(outVector.end(), row.begin(), row.end());
  }
  return totalTriangles;
}

float LoopMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field) {
  const Vec3_t<float> *pPoints = field.getPoints().data();
  const unsigned count = unsigned(field.getPoints().size());
  float value = std::numeric_limits<float>::max();
  for (unsigned i = 0; i < count; ++i) {
    float distanceSquared
      = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x)
      + (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y)
      + (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);
    value = std::min(value, distanceSquared);
  }
  return sqrt(value);
}

void LoopMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle) {
  (*mTriangles)[omp_get_thread_num()].push_back(triangle);
}

const BaseMeshBuilder::Triangle_t *LoopMeshBuilder::getTrianglesArray() const {
  return outVector.data();
}
