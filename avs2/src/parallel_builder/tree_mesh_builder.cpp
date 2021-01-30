/**
 * @file    tree_mesh_builder.cpp
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 * @author  Jakub Zárybnický <xzaryb00@stud.fit.vutbr.cz>
 * @date    20.12.2020
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <omp.h>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
  : BaseMeshBuilder(gridEdgeSize, "Octree") { }

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field) {
  unsigned triangles = 0;
  #pragma omp parallel
  #pragma omp single
  triangles = octreeStep(Vec3_t<float>(0, 0, 0), mGridSize / 2, field);
  return triangles;
}

unsigned TreeMeshBuilder::octreeStep(const Vec3_t<float> &start, unsigned a,
                                    const ParametricScalarField &field) {
  unsigned totalTriangles = 0;
  if (a == 1) {
    for (int i = 0; i < 8; i++) {
      Vec3_t<float> offset(start.x + sc_vertexNormPos[i].x * a,
                           start.y + sc_vertexNormPos[i].y * a,
                           start.z + sc_vertexNormPos[i].z * a);
      Vec3_t<float> center((offset.x + a / 2.) * mGridResolution,
                           (offset.y + a / 2.) * mGridResolution,
                           (offset.z + a / 2.) * mGridResolution);
      if (evaluateFieldAt(center, field) <= mIsoLevel + sqrt(3.) / 2 * a * mGridResolution) {
        totalTriangles += buildCube(offset, field);
      }
      return totalTriangles;
    }
  }
  for (int i = 0; i < 8; i++) {
    #pragma omp task shared(totalTriangles)
    {
      Vec3_t<float> offset(start.x + sc_vertexNormPos[i].x * a,
                           start.y + sc_vertexNormPos[i].y * a,
                           start.z + sc_vertexNormPos[i].z * a);
      Vec3_t<float> center((offset.x + a / 2.) * mGridResolution,
                           (offset.y + a / 2.) * mGridResolution,
                           (offset.z + a / 2.) * mGridResolution);
      if (evaluateFieldAt(center, field) <= mIsoLevel + sqrt(3.) / 2 * a * mGridResolution) {
        totalTriangles += octreeStep(offset, a / 2, field);
      }
    }
  }
  #pragma omp taskwait
  return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field) {
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

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle) {
  #pragma omp critical
  mTriangles.push_back(triangle);
}

const BaseMeshBuilder::Triangle_t *TreeMeshBuilder::getTrianglesArray() const {
  return mTriangles.data();
}
